"""Evaluation metrics (Sec. 3.11-3.12): DSC, HD95, ASD, PO-ROI, centroid error."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(pred, gt).sum() / s)


def _bbox(mask: np.ndarray, margin: int = 2):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)


def surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]):
    """Directed surface distances pred->gt and gt->pred in mm (None if a mask is empty)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return None, None
    sl = _bbox(pred | gt)
    p, g = pred[sl], gt[sl]
    sp, sg = _surface(p), _surface(g)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return dt_g[sp], dt_p[sg]


def hd95(pred, gt, spacing) -> float:
    a, b = surface_distances(pred, gt, spacing)
    if a is None:
        return 0.0 if (not pred.any() and not gt.any()) else float("nan")
    return float(max(np.percentile(a, 95), np.percentile(b, 95)))


def asd(pred, gt, spacing) -> float:
    a, b = surface_distances(pred, gt, spacing)
    if a is None:
        return 0.0 if (not pred.any() and not gt.any()) else float("nan")
    return float((a.sum() + b.sum()) / (len(a) + len(b)))


def po_roi(orbit_gt: np.ndarray, factor: float, min_fraction: float = 0.05) -> np.ndarray:
    """Peri-orbital ROI: each orbital-content component scaled by ``factor`` about its centroid.

    Used for evaluation only (oracle ROI built from the ground-truth orbital content).
    """
    orbit_gt = orbit_gt.astype(bool)
    roi = np.zeros_like(orbit_gt)
    lab, n = ndimage.label(orbit_gt)
    if n == 0:
        return roi
    sizes = ndimage.sum(orbit_gt, lab, range(1, n + 1))
    for k in range(1, n + 1):
        if sizes[k - 1] < min_fraction * sizes.max():
            continue
        comp = lab == k
        c = np.array(ndimage.center_of_mass(comp))
        idx = np.argwhere(comp)
        lo, hi = idx.min(0), idx.max(0)
        half = (hi - lo) / 2.0 * factor + 2
        a = np.maximum(np.floor(c - half), 0).astype(int)
        b = np.minimum(np.ceil(c + half) + 1, comp.shape).astype(int)
        grid = np.stack(np.meshgrid(*[np.arange(x, y) for x, y in zip(a, b)], indexing="ij"), 0).astype(np.float32)
        src = c[:, None, None, None] + (grid - c[:, None, None, None]) / factor
        inside = ndimage.map_coordinates(comp.astype(np.uint8), src.reshape(3, -1), order=0, mode="constant", cval=0)
        roi[a[0]:b[0], a[1]:b[1], a[2]:b[2]] |= inside.reshape(grid.shape[1:]).astype(bool)
    return roi


def centroid_error_mm(pred: np.ndarray, gt: np.ndarray, affine: np.ndarray) -> float:
    if not pred.any() or not gt.any():
        return float("nan")
    cp = np.append(ndimage.center_of_mass(pred.astype(bool)), 1.0)
    cg = np.append(ndimage.center_of_mass(gt.astype(bool)), 1.0)
    return float(np.linalg.norm((affine @ cp)[:3] - (affine @ cg)[:3]))


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray, spacing, roi: np.ndarray | None = None) -> dict:
    if roi is not None:
        pred, gt = pred & roi, gt & roi
    return {"dsc": dice(pred, gt), "hd95": hd95(pred, gt, spacing), "asd": asd(pred, gt, spacing)}
