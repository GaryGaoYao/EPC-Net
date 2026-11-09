"""End-to-end EPC-Net inference: CT -> Stage I orbital content -> priors -> Stage II bone.

Usage from Python::

    from epcnet.inference import EPCNetPredictor
    p = EPCNetPredictor("weights/stage1_eye.pth", "weights/stage2_bone.pth")
    res = p.predict("case_0000.nii.gz")        # dict with 'bone', 'orbit' (original grid), 'affine', 'timings'
    p.predict_to_files("case_0000.nii.gz", "out/", stl=True)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference

from .config import load_config
from .model import load_model
from .preprocessing import (load_volume, prepare_stage_input, resample_to_grid, resample_to_reference,
                            save_mask_like, to_ras)
from .priors import distance_to_mask, guidance_map

log = logging.getLogger("epcnet")


def _plain(t: torch.Tensor) -> torch.Tensor:
    return t.as_tensor() if isinstance(t, MetaTensor) else t


def metatensor_from_ijk(array_ijk: np.ndarray, ijk_to_ras: np.ndarray) -> MetaTensor:
    """Volume as (i, j, k) array + 4x4 IJK->RAS matrix (e.g. from a 3D Slicer volume node) -> model input."""
    data = torch.from_numpy(np.ascontiguousarray(array_ijk, dtype=np.float32))[None]
    return MetaTensor(data, affine=torch.as_tensor(np.asarray(ijk_to_ras, dtype=np.float64)))


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def sliding_probability(net, x: torch.Tensor, roi, device, sw_batch_size=2, overlap=0.5, mode="gaussian",
                        amp=True, out_device=None) -> torch.Tensor:
    """Foreground probability (X, Y, Z) on ``out_device`` for a (1, C, X, Y, Z) input."""
    out_device = out_device or device
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        logits = sliding_window_inference(x, roi_size=list(roi), sw_batch_size=int(sw_batch_size),
                                          predictor=lambda t: net(t.float()), overlap=float(overlap), mode=mode,
                                          sw_device=device, device=out_device)
    return torch.softmax(logits.float(), dim=1)[0, 1]


def stage1_prob_to_stage2_distance(prob1: np.ndarray, affine1, affine2, shape2, spacing2, threshold=0.5, cap=60.0):
    """Map the Stage-I probability onto the Stage-II grid, binarise (>0.5) and compute the distance map.

    No connected-component filtering or hole filling is applied (Sec. 3.7)."""
    mt = MetaTensor(torch.from_numpy(np.asarray(prob1, dtype=np.float32))[None], affine=torch.as_tensor(np.asarray(affine1)))
    prob2 = resample_to_grid(mt, affine2, shape2, mode="bilinear")[0].numpy()
    mask2 = prob2 > threshold
    return distance_to_mask(mask2, spacing2, cap), mask2


class EPCNetPredictor:
    def __init__(self, stage1_weights, stage2_weights, config: dict | str | None = None, device="auto",
                 overlap: float | None = None, sw_batch_size: int | None = None, amp: bool | None = None,
                 roi_margin_mm: float | None = None):
        """``roi_margin_mm``: if set, Stage II only runs inside the bounding box of the Stage-I orbital content
        enlarged by this margin (much faster, bone farther away is not segmented).  None = whole volume
        (as in the paper)."""
        self.roi_margin_mm = roi_margin_mm
        self.device = resolve_device(device)
        self.cfg = config if isinstance(config, dict) else load_config(config)
        self.net1, meta1 = load_model(stage1_weights, self.cfg["model"], self.device)
        self.net2, meta2 = load_model(stage2_weights, self.cfg["model"], self.device)
        # geometry/prior settings stored with the weights take precedence over the config
        self.spacing1 = (meta1.get("spacing") or self.cfg["stage1"]["spacing"])
        self.patch1 = (meta1.get("patch_size") or self.cfg["stage1"]["patch_size"])
        self.spacing2 = (meta2.get("spacing") or self.cfg["stage2"]["spacing"])
        self.patch2 = (meta2.get("patch_size") or self.cfg["stage2"]["patch_size"])
        self.prior = meta2.get("prior") or self.cfg["prior"]
        self.preprocess = meta1.get("preprocess") or self.cfg["preprocess"]
        self.use_guidance = self.net2.convInit.conv.in_channels == 2
        inf = self.cfg["inference"]
        self.overlap = float(overlap if overlap is not None else inf["sw_overlap"])
        self.sw_batch_size = int(sw_batch_size or inf["sw_batch_size"])
        self.mode = inf.get("sw_mode", "gaussian")
        self.threshold = float(inf["threshold"])
        self.amp = bool(inf["amp"] if amp is None else amp)

    def _sw(self, net, x, roi):
        try:
            return sliding_probability(net, x, roi, self.device, self.sw_batch_size, self.overlap, self.mode, self.amp)
        except torch.cuda.OutOfMemoryError:
            log.warning("CUDA OOM - retrying with full-volume aggregation on CPU")
            torch.cuda.empty_cache()
            return sliding_probability(net, x, roi, self.device, 1, self.overlap, self.mode, self.amp,
                                       out_device=torch.device("cpu"))

    def _roi(self, mask2: np.ndarray):
        if self.roi_margin_mm is None or not mask2.any():
            return None
        idx = np.argwhere(mask2)
        m = [int(np.ceil(float(self.roi_margin_mm) / float(s))) for s in self.spacing2]
        lo = np.maximum(idx.min(0) - m, 0)
        hi = np.minimum(idx.max(0) + 1 + np.array(m), mask2.shape)
        return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    @torch.no_grad()
    def predict(self, image, progress=None) -> dict:
        """``image``: file path, DICOM folder or channel-first MetaTensor (RAS affine).
        ``progress``: optional callable(fraction: float, message: str)."""
        report = progress or (lambda f, m: None)
        t = {"start": time.perf_counter()}
        report(0.02, "Loading image")
        orig = image if isinstance(image, MetaTensor) else load_volume(image)
        ras = to_ras(orig, self.preprocess.get("orientation", "RAS"))
        pct = self.preprocess["percentiles"]
        t["load"] = time.perf_counter()

        # ---- Stage I: orbital content at 1 mm
        report(0.08, "Stage I: orbital content")
        x1, meta1 = prepare_stage_input(ras, self.spacing1, pct)
        prob1 = _plain(self._sw(self.net1, x1[None].to(self.device), self.patch1).cpu())
        t["stage1"] = time.perf_counter()

        # ---- priors on the Stage II grid
        report(0.30, "Computing orbital-content priors")
        x2, meta2 = prepare_stage_input(ras, self.spacing2, pct)
        dist, mask2 = stage1_prob_to_stage2_distance(prob1.numpy(), meta1["affine"], meta2["affine"], meta2["shape"],
                                                 self.spacing2, self.prior.get("threshold", 0.5), self.prior["dist_cap"])
        orbit_found = bool((dist <= 0).any())
        if not orbit_found:
            log.warning("Stage I found no orbital content - guidance map is empty.")
        inp = x2[None]
        if self.use_guidance:
            g = guidance_map(torch.from_numpy(dist), self.prior["sigma"], self.prior["delta"])
            inp = torch.cat([inp, g[None, None]], 1)
        t["prior"] = time.perf_counter()

        # ---- Stage II: orbital bone at 0.5 x 0.5 x 1.0 mm (whole volume or orbital region)
        report(0.40, "Stage II: orbital bone")
        roi = self._roi(mask2)
        if roi is None:
            prob2 = _plain(self._sw(self.net2, inp.to(self.device), self.patch2).cpu())
        else:
            # compute on the ROI plus half a patch of context, keep only the ROI
            ctx = tuple(slice(max(0, r.start - p // 2), min(n, r.stop + p // 2))
                        for r, p, n in zip(roi, self.patch2, inp.shape[2:]))
            sub = _plain(self._sw(self.net2, inp[(slice(None), slice(None)) + ctx].contiguous().to(self.device),
                                  self.patch2).cpu())
            prob2 = torch.zeros(inp.shape[2:], dtype=torch.float32)
            prob2[roi] = sub[tuple(slice(r.start - c.start, r.stop - c.start) for r, c in zip(roi, ctx))]
        t["stage2"] = time.perf_counter()
        report(0.85, "Resampling to the original image grid")

        # ---- back to the original image grid
        p2 = MetaTensor(prob2[None], affine=torch.as_tensor(np.asarray(meta2["affine"])))
        p1 = MetaTensor(prob1[None], affine=torch.as_tensor(np.asarray(meta1["affine"])))
        bone = resample_to_reference(p2, orig, mode="bilinear")[0].numpy() > self.threshold
        orbit = resample_to_reference(p1, orig, mode="bilinear")[0].numpy() > self.prior.get("threshold", 0.5)
        t["resample"] = time.perf_counter()
        timings = {k: round(t[k] - t[p], 3) for p, k in zip(["start", "load", "stage1", "prior", "stage2"],
                                                           ["load", "stage1", "prior", "stage2", "resample"])}
        timings["total"] = round(t["resample"] - t["start"], 3)
        report(1.0, "Done")
        return {"bone": bone.astype(np.uint8), "orbit": orbit.astype(np.uint8),
                "affine": np.asarray(orig.affine, dtype=np.float64), "timings": timings, "orbit_found": orbit_found}

    def predict_to_files(self, image_path, out_dir, case_id: str | None = None, save_orbit=True, combined=True,
                         stl: bool = False) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image_path)
        cid = case_id or image_path.name.replace(".nii.gz", "").replace(".nii", "").removesuffix("_0000")
        res = self.predict(image_path)
        files = {}
        save_mask_like(res["bone"], image_path, res["affine"], out_dir / f"{cid}_bone.nii.gz")
        files["bone"] = str(out_dir / f"{cid}_bone.nii.gz")
        if save_orbit:
            save_mask_like(res["orbit"], image_path, res["affine"], out_dir / f"{cid}_orbit.nii.gz")
            files["orbit"] = str(out_dir / f"{cid}_orbit.nii.gz")
        if combined:  # same label convention as the training data: 1 = bone, 2 = orbital content
            lab = np.where(res["bone"] > 0, 1, np.where(res["orbit"] > 0, 2, 0)).astype(np.uint8)
            save_mask_like(lab, image_path, res["affine"], out_dir / f"{cid}.nii.gz")
            files["labels"] = str(out_dir / f"{cid}.nii.gz")
        if stl and not res["bone"].any():
            log.warning("%s: empty bone mask - STL skipped", cid)
        elif stl:
            from .surface import mask_to_stl

            mask_to_stl(res["bone"], res["affine"], out_dir / f"{cid}_bone.stl", self.cfg["surface"])
            files["stl"] = str(out_dir / f"{cid}_bone.stl")
        res["files"] = files
        (out_dir / f"{cid}_timing.json").write_text(json.dumps(res["timings"], indent=1))
        return res
