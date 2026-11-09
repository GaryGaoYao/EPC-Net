"""Glue between stages: Stage-I prior generation for the training cache, evaluation, release export."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch

from . import __version__
from . import config as C
from .data import discover_cases, load_split, resolve_label
from .inference import EPCNetPredictor, resolve_device, sliding_probability, stage1_prob_to_stage2_distance
from .metrics import centroid_error_mm, dice, hd95, po_roi, segmentation_metrics
from .model import extract_state_dict, load_model, load_torch
from .preprocessing import load_volume, read_meta, save_mask_like, spacing_from_affine
from .priors import band_coordinates
from .data import BAND_STORE_MM

log = logging.getLogger("epcnet")


# ------------------------------------------------------------------ priors
@torch.no_grad()
def generate_priors(cfg: dict, stage1_ckpt: Path, case_ids=None, device="auto") -> None:
    """Run the trained Stage-I model on every cached case and store d_raw on the Stage-II grid."""
    dev = resolve_device(device)
    net, meta = load_model(stage1_ckpt, cfg["model"], dev)
    cache = C.cache_dir(cfg)
    s1, s2 = cache / "stage1", cache / "stage2"
    ids = case_ids or sorted(p.name[:-5] for p in s1.glob("*.json"))
    inf = cfg["inference"]
    rows = []
    for cid in ids:
        t0 = time.time()
        m1, m2 = read_meta(s1 / f"{cid}.json"), read_meta(s2 / f"{cid}.json")
        x = torch.from_numpy(np.load(s1 / f"{cid}_img.npy").astype(np.float32))[None, None].to(dev)
        prob = sliding_probability(net, x, cfg["stage1"]["patch_size"], dev, inf["sw_batch_size"], inf["sw_overlap"],
                                   inf["sw_mode"], inf["amp"]).cpu().numpy()
        d, mask2 = stage1_prob_to_stage2_distance(prob, m1["affine"], m2["affine"], m2["shape"], m2["spacing"],
                                                  cfg["prior"]["threshold"], cfg["prior"]["dist_cap"])
        np.save(s2 / f"{cid}_dist_pred.npy", d.astype(np.float16))
        coords, dist = band_coordinates(d, BAND_STORE_MM)
        np.savez(s2 / f"{cid}_band_pred.npz", coords=coords, dist=dist)
        gt = np.load(s2 / f"{cid}_orbit.npy").astype(bool)
        dsc = dice(mask2, gt)
        ce = centroid_error_mm(mask2, gt, np.asarray(m2["affine"]))
        rows.append({"case_id": cid, "orbit_dsc": dsc, "centroid_error_mm": ce})
        log.info("priors %-20s Stage-I DSC %.4f | centroid err %.2f mm | %.1fs", cid, dsc, ce, time.time() - t0)
    with open(s2 / "stage1_prior_qc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------- evaluation
def evaluate(cfg: dict, stage1_ckpt: Path, stage2_ckpt: Path, split_name: str = "test", out_dir: Path | None = None,
             save_predictions: bool = False, device="auto", case_ids=None) -> dict:
    """End-to-end evaluation on raw (original-resolution) images: global + PO-ROI DSC/HD95/ASD."""
    raw = C.raw_dir(cfg)
    split = load_split(cfg)
    ids = case_ids or split.get(split_name) or []
    if not ids:
        raise ValueError(f"Split '{split_name}' is empty.")
    cases = {c["case_id"]: c for c in discover_cases(raw)}
    bone_v, orbit_v = resolve_label(raw, cfg["data"]["bone_label"]), resolve_label(raw, cfg["data"]["orbit_label"])
    out_dir = Path(out_dir or C.run_dir(cfg, 2).parent / f"eval_{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = EPCNetPredictor(stage1_ckpt, stage2_ckpt, cfg, device=device)
    factors = cfg["evaluation"]["po_roi_factors"]
    rows = []
    for cid in ids:
        c = cases[cid]
        res = pred.predict(c["image"])
        lbl = load_volume(c["label"])[0].numpy().round().astype(np.int16)
        gt_bone, gt_orbit = lbl == bone_v, lbl == orbit_v
        sp = spacing_from_affine(res["affine"])
        row = {"case_id": cid, "seconds": res["timings"]["total"]}
        for k, v in segmentation_metrics(res["bone"].astype(bool), gt_bone, sp).items():
            row[f"global_{k}"] = v
        for f in factors:
            roi = po_roi(gt_orbit, f)
            for k, v in segmentation_metrics(res["bone"].astype(bool), gt_bone, sp, roi).items():
                row[f"poroi{f:.1f}_{k}"] = v
        row["stage1_dsc"] = dice(res["orbit"], gt_orbit)
        row["stage1_hd95"] = hd95(res["orbit"].astype(bool), gt_orbit, sp)
        row["stage1_centroid_err_mm"] = centroid_error_mm(res["orbit"], gt_orbit, res["affine"])
        rows.append(row)
        log.info("eval %-18s DSC %.4f HD95 %.2f ASD %.2f | PO-ROI1.1 DSC %.4f | Stage-I DSC %.3f | %.1fs", cid,
                 row["global_dsc"], row["global_hd95"], row["global_asd"], row.get("poroi1.1_dsc", np.nan),
                 row["stage1_dsc"], row["seconds"])
        if save_predictions:
            save_mask_like(res["bone"], c["image"], res["affine"], out_dir / f"{cid}_bone.nii.gz")
            save_mask_like(res["orbit"], c["image"], res["affine"], out_dir / f"{cid}_orbit.nii.gz")
    with open(out_dir / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nan-means for empty ROIs
    summary = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "sd": float(np.nanstd([r[k] for r in rows]))}
               for k in rows[0] if k != "case_id"}
    summary["n_cases"] = len(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Global DSC %.4f ± %.4f | HD95 %.2f ± %.2f mm | ASD %.2f ± %.2f mm  -> %s",
             summary["global_dsc"]["mean"], summary["global_dsc"]["sd"], summary["global_hd95"]["mean"],
             summary["global_hd95"]["sd"], summary["global_asd"]["mean"], summary["global_asd"]["sd"], out_dir)
    return summary


# ------------------------------------------------------------------ export
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _access_terms(rel: dict, version: str) -> str:
    contacts = "\n".join(f"  - {c}" for c in rel.get("contacts", []))
    return f"""# {rel.get('weights_name', 'EPC-Net weights')} v{version}

Training data: {rel.get('training_data', 'see weights/README.md')}
Access: {rel.get('access', 'restricted')}

These weights are provided only to applicants whose signed "Model Weight Access Request and
Research-Use Statement" has been approved in writing by the corresponding authors:
{contacts}
  (cc: {rel.get('cc', '')})

Conditions of use:
  - research use only; not for clinical decision-making; not a certified medical device
  - no redistribution of the weights to third parties
  - comply with the licence / usage terms of the public training datasets
  - cite: Gao Y. et al., Biomed. Signal Process. Control 130 (2027) 111465, doi:10.1016/j.bspc.2026.111465

The original weights reported in the paper were trained on clinical data of University Hospitals
Leuven / KU Leuven (Ethics Committee approval S68944). They are not part of this package; whether
they can be shared is determined by the hospital and its Ethics Committee.
"""


def export_release(cfg: dict, stage1_ckpt: Path, stage2_ckpt: Path, out_dir: Path, version: str | None = None,
                   make_zip: bool = True) -> Path:
    """Write a weights-only release package (no optimiser state, no patient identifiers).

    ``stage1_eye.pth`` / ``stage2_bone.pth`` keep the file names expected by the EPC-Net GUI.
    """
    version = version or __version__
    out_dir = Path(out_dir)
    wdir = out_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": "EPC-Net", "version": version, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "paper": "doi:10.1016/j.bspc.2026.111465", "files": {}}
    for name, ckpt in (("stage1_eye.pth", stage1_ckpt), ("stage2_bone.pth", stage2_ckpt)):
        obj = load_torch(ckpt)
        sd = {k: v.detach().float().cpu() for k, v in extract_state_dict(obj).items()}
        meta = dict(obj.get("meta", {})) if isinstance(obj, dict) else {}
        for k in ("train_cases", "val_cases", "val_per_case"):  # strip case identifiers from the release
            meta.pop(k, None)
        meta = json.loads(json.dumps(meta, default=str))  # plain types only -> loads with weights_only=True
        torch.save({"state_dict": sd, "meta": meta}, wdir / name)
        manifest["files"][name] = {"sha256": _sha256(wdir / name), "meta": meta}
    rel = cfg.get("release", {})
    manifest.update({"weights_name": rel.get("weights_name"), "training_data": rel.get("training_data"),
                     "access": rel.get("access"), "contacts": rel.get("contacts", []), "cc": rel.get("cc")})
    (out_dir / "ACCESS_TERMS.md").write_text(_access_terms(rel, version), encoding="utf-8")
    run_cfg = {k: cfg[k] for k in ("preprocess", "model", "stage1", "stage2", "prior", "inference", "surface")}
    C.save_config(run_cfg, out_dir / "epcnet_config.yaml")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if make_zip:
        zpath = out_dir.parent / f"{out_dir.name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out_dir.parent))
        log.info("Release package: %s", zpath)
        return zpath
    return out_dir
