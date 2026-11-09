"""Orbital-content-derived anatomical priors (Sec. 3.8.1, Algorithm 1).

Given the (predicted) orbital-content mask E, the mask is dilated by ``delta`` mm and
the Euclidean distance d(p) to the dilated mask is computed.  Two maps follow:

    G(p) = exp(-d / sigma)                      feature prior   (2nd input channel)
    W(p) = 1 + alpha * exp(-d^2 / (2 sigma^2))  optimisation prior (loss weights)

Implementation note: for a Euclidean dilation with radius ``delta`` the distance to
the dilated mask equals ``max(d_raw - delta, 0)`` where ``d_raw`` is the distance to
the undilated mask.  We therefore cache ``d_raw`` once and derive G and W on the fly,
which lets sigma, alpha, delta and d_max be changed (e.g. for the OAT sensitivity
analysis of Sec. 3.15) without recomputing any distance transform.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def distance_to_mask(mask: np.ndarray, spacing: Sequence[float], cap: float = 60.0) -> np.ndarray:
    """Euclidean distance (mm) from every voxel to the nearest mask voxel, clipped at ``cap``.

    Only a bounding box around the mask (+cap mm) is transformed, which is exact for all
    voxels with distance < cap and an order of magnitude faster on full-resolution CT.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.full(mask.shape, cap, dtype=np.float32)
    if not mask.any():
        return out
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    margin = [int(np.ceil(cap / float(s))) + 1 for s in spacing]
    sl = tuple(slice(max(0, l - m), min(n, h + m)) for l, h, m, n in zip(lo, hi, margin, mask.shape))
    d = distance_transform_edt(~mask[sl], sampling=[float(s) for s in spacing]).astype(np.float32)
    out[sl] = np.minimum(d, cap)
    return out


def _dilated(d, delta: float):
    if isinstance(d, torch.Tensor):
        return torch.clamp(d - delta, min=0.0)
    return np.maximum(d - delta, 0.0)


def guidance_map(d_raw, sigma: float = 5.0, delta: float = 1.5):
    d = _dilated(d_raw, delta)
    return torch.exp(-d / sigma) if isinstance(d, torch.Tensor) else np.exp(-d / sigma).astype(np.float32)


def weight_map(d_raw, sigma: float = 5.0, alpha: float = 6.0, delta: float = 1.5):
    d = _dilated(d_raw, delta)
    if isinstance(d, torch.Tensor):
        return 1.0 + alpha * torch.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return (1.0 + alpha * np.exp(-(d ** 2) / (2.0 * sigma ** 2))).astype(np.float32)


def band_coordinates(d_raw: np.ndarray, max_mm: float, max_points: int = 300_000, seed: int = 0):
    """Voxel coordinates with d_raw <= max_mm (random subsample) and their distances.

    Used for peri-orbital focused patch sampling: at training time the coordinates are
    filtered with ``max(d_raw - delta, 0) <= d_max``.
    """
    coords = np.argwhere(d_raw <= max_mm)
    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        coords = coords[rng.choice(len(coords), max_points, replace=False)]
    dist = d_raw[tuple(coords.T)] if len(coords) else np.zeros(0, np.float32)
    return coords.astype(np.int32), dist.astype(np.float16)
