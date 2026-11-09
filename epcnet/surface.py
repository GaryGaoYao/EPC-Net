"""Volumetric-to-surface reconstruction (Sec. 3.9).

Gaussian smoothing (sigma = 1 voxel) -> Marching Cubes (iso 0.5) -> 20 iterations of
Laplacian smoothing (relaxation 0.01) -> binary STL.  Pure NumPy/SciPy/scikit-image,
no VTK needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage, sparse


def laplacian_smooth(verts: np.ndarray, faces: np.ndarray, iterations: int = 20, relaxation: float = 0.01) -> np.ndarray:
    """VTK-style Laplacian smoothing: v <- v + relaxation * (mean(neighbours) - v)."""
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    adj = sparse.coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    adj.data[:] = 1.0  # duplicate edges count once
    deg = np.asarray(adj.sum(1)).ravel()
    deg[deg == 0] = 1.0
    v = verts.astype(np.float64).copy()
    for _ in range(int(iterations)):
        v += relaxation * (adj @ v / deg[:, None] - v)
    return v


def mask_to_mesh(mask: np.ndarray, affine: np.ndarray, sigma_vox: float = 1.0, level: float = 0.5,
                 iterations: int = 20, relaxation: float = 0.01, coordinates: str = "LPS"):
    """Binary mask (voxel grid with RAS ``affine``) -> (vertices in mm, triangle faces)."""
    from skimage.measure import marching_cubes

    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        raise ValueError("Empty mask - nothing to reconstruct.")
    pad = int(np.ceil(4 * sigma_vox)) + 2
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + 1 + pad, mask.shape)
    sub = np.zeros(tuple(hi - lo + 2), np.float32)  # +1 border of zeros => closed surface
    sub[1:-1, 1:-1, 1:-1] = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if sigma_vox > 0:
        sub = ndimage.gaussian_filter(sub, sigma=float(sigma_vox))
    verts, faces, _, _ = marching_cubes(sub, level=float(level))
    verts = verts + lo - 1  # back to full-volume voxel indices
    verts = laplacian_smooth(verts, faces, iterations, relaxation)
    world = (np.c_[verts, np.ones(len(verts))] @ np.asarray(affine, dtype=np.float64).T)[:, :3]
    if coordinates.upper() == "LPS":
        world[:, :2] *= -1.0
    flip = np.linalg.det(np.asarray(affine)[:3, :3]) < 0
    faces = faces[:, ::-1] if flip else faces  # keep outward-facing normals
    return world.astype(np.float32), faces.astype(np.int64)


def write_stl(path: str | Path, verts: np.ndarray, faces: np.ndarray, header: str = "EPC-Net orbital bone") -> None:
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norm == 0, 1, norm)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = normals, tri
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii", "ignore")[:80].ljust(80, b" "))
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(rec.tobytes())


def mask_to_stl(mask: np.ndarray, affine: np.ndarray, path: str | Path, surface_cfg: dict | None = None) -> int:
    s = surface_cfg or {}
    v, f = mask_to_mesh(mask, affine, s.get("gaussian_sigma_vox", 1.0), s.get("iso_level", 0.5),
                        s.get("laplacian_iterations", 20), s.get("relaxation", 0.01), s.get("coordinates", "LPS"))
    write_stl(path, v, f)
    return len(f)
