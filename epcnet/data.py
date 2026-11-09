"""Dataset discovery (nnU-Net raw layout), patient-level splits, preprocessing cache and
patch datasets for both stages.

Cache layout (``data.cache_dir``)::

    stage1/<case>_img.npy      float16, normalised CT at 1.0 mm (RAS, x-y-z order)
    stage1/<case>_lbl.npy      uint8,   orbital content
    stage1/<case>.json         affine / shape / spacing
    stage2/<case>_img.npy      float16, normalised CT at 0.5 x 0.5 x 1.0 mm
    stage2/<case>_lbl.npy      uint8,   orbital bone
    stage2/<case>_orbit.npy    uint8,   ground-truth orbital content (for gt priors / QA)
    stage2/<case>_dist_gt.npy  float16, distance (mm) to GT orbital content
    stage2/<case>_dist_pred.npy float16, distance (mm) to Stage-I predicted orbital content
    stage2/<case>_band_{gt,pred}.npz  peri-orbital coordinates for focused sampling
"""

from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C
from .augment import enlarged_size
from .preprocessing import load_volume, prepare_stage_input, read_meta, resample_to_spacing, to_ras, write_meta
from .priors import band_coordinates, distance_to_mask, guidance_map

log = logging.getLogger("epcnet")
BAND_STORE_MM = 45.0  # coordinates up to this distance are stored; filtered by d_max at runtime
CACHE_VERSION = 2     # bump when preprocessing changes -> outdated cases are rebuilt automatically


# ------------------------------------------------------------------ discovery
def discover_cases(raw: Path) -> list[dict]:
    images = sorted((raw / "imagesTr").glob("*_0000.nii.gz")) or sorted((raw / "imagesTr").glob("*_0000.nii"))
    if not images:
        raise FileNotFoundError(f"No *_0000.nii.gz images in {raw / 'imagesTr'}")
    cases = []
    for img in images:
        cid = img.name.split("_0000.nii")[0]
        lbl = raw / "labelsTr" / (cid + ".nii.gz")
        if not lbl.exists():
            lbl = raw / "labelsTr" / (cid + ".nii")
        if not lbl.exists():
            raise FileNotFoundError(f"Missing label for case {cid}")
        cases.append({"case_id": cid, "image": str(img), "label": str(lbl)})
    return cases


def resolve_label(raw: Path, value) -> int:
    if isinstance(value, int):
        return value
    ds = json.loads((raw / "dataset.json").read_text(encoding="utf-8"))
    labels = ds["labels"]
    if value not in labels:
        raise KeyError(f"Label '{value}' not in dataset.json labels {labels}")
    v = labels[value]
    return int(v[0] if isinstance(v, (list, tuple)) else v)


# ---------------------------------------------------------------------- splits
def make_split(case_ids: Sequence[str], n_val: int, n_test: int, seed: int) -> dict:
    ids = sorted(case_ids)
    random.Random(seed).shuffle(ids)
    test, val, train = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]
    return {"seed": seed, "train": sorted(train), "val": sorted(val), "test": sorted(test)}


def split_from_nnunet(splits_final: Path, fold: int) -> dict:
    folds = json.loads(Path(splits_final).read_text(encoding="utf-8"))
    return {"source": str(splits_final), "fold": fold, "train": sorted(folds[fold]["train"]),
            "val": sorted(folds[fold]["val"]), "test": []}


def load_split(cfg: dict) -> dict:
    path = Path(cfg["data"]["split_file"])
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} not found - run `python -m epcnet split` first.")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------- preprocessing
def resample_label(lbl_ras, value: int, spacing) -> np.ndarray:
    """Binary class map resampled by linear interpolation + 0.5 threshold (nnU-Net style).

    Smoother and more faithful than nearest-neighbour for 1-2 voxel thin orbital walls."""
    onehot = (lbl_ras.round() == value).float()
    return resample_to_spacing(onehot, spacing, mode="bilinear")[0].numpy() > 0.5


def _preprocess_one(args) -> str:
    case, cfg, bone_value, orbit_value = args
    cache = C.cache_dir(cfg)
    cid = case["case_id"]
    s1, s2 = cache / "stage1", cache / "stage2"
    if _cache_ok(s1, s2, cid):
        return f"{cid}: cached"
    t0 = time.time()
    pct = cfg["preprocess"]["percentiles"]
    img = to_ras(load_volume(case["image"]), cfg["preprocess"]["orientation"])
    lbl = to_ras(load_volume(case["label"]), cfg["preprocess"]["orientation"])

    # Stage I (1 mm): CT + orbital content
    x1, meta1 = prepare_stage_input(img, cfg["stage1"]["spacing"], pct)
    np.save(s1 / f"{cid}_img.npy", x1[0].numpy().astype(np.float16))
    np.save(s1 / f"{cid}_lbl.npy", resample_label(lbl, orbit_value, cfg["stage1"]["spacing"]).astype(np.uint8))
    meta1["cache_version"] = CACHE_VERSION
    write_meta(s1 / f"{cid}.json", meta1)
    del x1

    # Stage II (0.5 x 0.5 x 1.0 mm): CT + bone + GT orbital content
    x2, meta2 = prepare_stage_input(img, cfg["stage2"]["spacing"], pct)
    np.save(s2 / f"{cid}_img.npy", x2[0].numpy().astype(np.float16))
    del x2
    np.save(s2 / f"{cid}_lbl.npy", resample_label(lbl, bone_value, cfg["stage2"]["spacing"]).astype(np.uint8))
    orbit = resample_label(lbl, orbit_value, cfg["stage2"]["spacing"])
    np.save(s2 / f"{cid}_orbit.npy", orbit.astype(np.uint8))
    d = distance_to_mask(orbit, cfg["stage2"]["spacing"], cfg["prior"]["dist_cap"])
    np.save(s2 / f"{cid}_dist_gt.npy", d.astype(np.float16))
    coords, dist = band_coordinates(d, BAND_STORE_MM)
    np.savez(s2 / f"{cid}_band_gt.npz", coords=coords, dist=dist)
    meta2["cache_version"] = CACHE_VERSION
    write_meta(s2 / f"{cid}.json", meta2)  # written last: marks the case as complete
    return f"{cid}: stage1 {meta1['shape']} stage2 {meta2['shape']} ({time.time() - t0:.0f}s)"


def _cache_ok(s1: Path, s2: Path, cid: str) -> bool:
    try:
        return all(read_meta(d / f"{cid}.json").get("cache_version") == CACHE_VERSION for d in (s1, s2)) \
            and (s2 / f"{cid}_band_gt.npz").exists()
    except (FileNotFoundError, ValueError):
        return False


def preprocess_dataset(cfg: dict, case_ids: Sequence[str] | None = None) -> None:
    raw = C.raw_dir(cfg)
    cases = discover_cases(raw)
    if case_ids:
        cases = [c for c in cases if c["case_id"] in set(case_ids)]
    bone = resolve_label(raw, cfg["data"]["bone_label"])
    orbit = resolve_label(raw, cfg["data"]["orbit_label"])
    cache = C.cache_dir(cfg)
    (cache / "stage1").mkdir(parents=True, exist_ok=True)
    (cache / "stage2").mkdir(parents=True, exist_ok=True)
    log.info("Preprocessing %d cases from %s (bone=%d, orbital content=%d) -> %s", len(cases), raw, bone, orbit, cache)
    jobs = [(c, cfg, bone, orbit) for c in cases]
    todo = [j for j in jobs if not _cache_ok(cache / "stage1", cache / "stage2", j[0]["case_id"])]
    log.info("%d cases up to date, %d to (re)build", len(jobs) - len(todo), len(todo))
    workers = min(int(cfg["data"].get("preprocess_workers", 1)), max(len(todo), 1))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            futures = {ex.submit(_preprocess_one, j): j[0]["case_id"] for j in todo}
            for k, fut in enumerate(as_completed(futures), 1):
                log.info("[%d/%d] %s", k, len(todo), fut.result())
    else:
        for k, job in enumerate(todo, 1):
            log.info("[%d/%d] %s", k, len(todo), _preprocess_one(job))


# --------------------------------------------------------------- crop helpers
def crop_with_pad(arr: np.ndarray, center: Sequence[int], size: Sequence[int], pad_value: float) -> np.ndarray:
    """Crop ``size`` around ``center`` from a (possibly memory-mapped) 3D array, padding outside."""
    out = np.full(tuple(size), pad_value, dtype=np.float32)
    src, dst = [], []
    for c, s, n in zip(center, size, arr.shape):
        start = int(c) - s // 2
        a0, a1 = max(start, 0), min(start + s, n)
        src.append(slice(a0, a1))
        dst.append(slice(a0 - start, a1 - start))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _uniform_center(shape, size, rng) -> list[int]:
    c = []
    for n, s in zip(shape, size):
        lo, hi = s // 2, n - (s - s // 2)
        c.append(int(rng.integers(lo, hi + 1)) if hi >= lo else n // 2)
    return c


# ------------------------------------------------------------- patch dataset
class PatchDataset(Dataset):
    """Random, slightly enlarged training crops from the memory-mapped cache.

    Augmentation and the final crop happen on the GPU (``epcnet.augment.GpuAugment``).
    Stage 1: random centres.  Stage 2: with probability ``focus_prob`` the centre is drawn from
    the peri-orbital band {p : max(d_raw - delta, 0) <= d_max} (Algorithm 1, line 15).
    One epoch = ``iterations_per_epoch`` x ``batch_size`` patches, cycling over all training cases.
    """

    def __init__(self, cfg: dict, stage: int, case_ids: Sequence[str]):
        self.cfg = cfg
        self.stage = stage
        sc = C.stage_config(cfg, stage)
        self.dir = C.cache_dir(cfg) / f"stage{stage}"
        self.case_ids = list(case_ids)
        self.patch = [int(v) for v in sc["patch_size"]]
        self.big = enlarged_size(self.patch, cfg["augment"]["crop_margin"])
        self.length = int(sc["iterations_per_epoch"]) * int(sc["batch_size"])
        self.prior = cfg["prior"]
        self.focus_prob = float(sc.get("focus_prob", 0.0)) if stage == 2 else 0.0
        self.src = self.prior.get("source", "pred")
        self._arrays: dict = {}
        self._bands: dict = {}
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        if stage == 2:
            for cid in self.case_ids:
                p = self.dir / f"{cid}_dist_{self.src}.npy"
                if not p.exists():
                    raise FileNotFoundError(f"{p} missing - run `python -m epcnet priors` (or set prior.source=gt).")

    def __getstate__(self):  # do not pickle open memmaps into DataLoader workers
        state = self.__dict__.copy()
        state["_arrays"], state["_bands"] = {}, {}
        return state

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def _get(self, cid: str) -> dict:
        if cid not in self._arrays:
            a = {"image": np.load(self.dir / f"{cid}_img.npy", mmap_mode="r"),
                 "label": np.load(self.dir / f"{cid}_lbl.npy", mmap_mode="r")}
            if self.stage == 2:
                a["dist"] = np.load(self.dir / f"{cid}_dist_{self.src}.npy", mmap_mode="r")
                band = np.load(self.dir / f"{cid}_band_{self.src}.npz")
                d = np.maximum(band["dist"].astype(np.float32) - self.prior["delta"], 0.0)
                self._bands[cid] = band["coords"][d <= self.prior["d_max"]]
            self._arrays[cid] = a
        return self._arrays[cid]

    def __getitem__(self, idx: int) -> dict:
        cid = self.case_ids[idx % len(self.case_ids)]
        a = self._get(cid)
        band = self._bands.get(cid)
        if self.stage == 2 and band is not None and len(band) and self.rng.random() < self.focus_prob:
            center = band[self.rng.integers(len(band))].tolist()
        else:
            center = _uniform_center(a["image"].shape, self.patch, self.rng)
        out = {"image": torch.from_numpy(crop_with_pad(a["image"], center, self.big, 0.0)[None]),
               "label": torch.from_numpy(crop_with_pad(a["label"], center, self.big, 0.0)[None].astype(np.uint8))}
        if self.stage == 2:
            out["dist"] = torch.from_numpy(crop_with_pad(a["dist"], center, self.big, self.prior["dist_cap"])[None])
        return out


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.set_seed(int(info.seed % (2 ** 32)))


# ------------------------------------------------------------ full volumes
def load_case_volume(cfg: dict, stage: int, cid: str, prior_source: str | None = None) -> dict:
    """Full preprocessed volume for validation: network input (1,C,X,Y,Z), label, spacing."""
    d = C.cache_dir(cfg) / f"stage{stage}"
    sc = C.stage_config(cfg, stage)
    img = torch.from_numpy(np.load(d / f"{cid}_img.npy").astype(np.float32))[None, None]
    lbl = torch.from_numpy(np.load(d / f"{cid}_lbl.npy"))
    if stage == 2 and sc.get("use_guidance", True):
        src = prior_source or cfg["prior"]["source"]
        dist = torch.from_numpy(np.load(d / f"{cid}_dist_{src}.npy").astype(np.float32))[None, None]
        img = torch.cat([img, guidance_map(dist, cfg["prior"]["sigma"], cfg["prior"]["delta"])], 1)
    meta = read_meta(d / f"{cid}.json")
    return {"image": img, "label": lbl, "spacing": meta["spacing"], "case_id": cid}
