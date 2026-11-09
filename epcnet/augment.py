"""GPU data augmentation (Sec. 3.5), applied batch-wise on the training device.

The DataLoader only cuts slightly enlarged crops from the memory-mapped cache; flipping,
rotation, isotropic scaling and the final crop are done here in a single ``grid_sample``
on the GPU, so the GPU is not starved by CPU-side 3D interpolation.

Spatial (identical for CT, label and orbital-content distance map):
    left-right flip p=0.5 | rotation +-10 deg about each axis p=0.3 | isotropic scale 0.9-1.1 p=0.3
Intensity (CT only):
    shift +-0.10 | contrast scaling 0.9-1.1 | Gaussian noise sigma in [0, 0.05]

Rotations and scaling are performed in physical (mm) space, so they stay rigid on the
anisotropic 0.5 x 0.5 x 1.0 mm Stage-II grid.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def enlarged_size(patch: Sequence[int], margin: float) -> list[int]:
    """Crop size before augmentation; (big - patch) is even so the identity transform is an exact centre crop."""
    return [int(p + 2 * math.ceil(p * (margin - 1.0) / 2.0)) for p in patch]


def _rot(axis: int, a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    if axis == 0:
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == 1:
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return torch.stack([torch.stack(r, -1) for r in m], -2)  # (B, 3, 3)


class GpuAugment:
    def __init__(self, aug: dict, patch: Sequence[int], big: Sequence[int], spacing: Sequence[float]):
        self.a = aug
        self.patch = [int(p) for p in patch]
        self.big = [int(b) for b in big]
        self.spacing = [float(s) for s in spacing]

    def _theta(self, b: int, device, train: bool) -> torch.Tensor:
        a = self.a
        eye = torch.eye(3, device=device).expand(b, 3, 3)
        if not train:
            m = eye
        else:
            u = lambda: torch.rand(b, device=device)  # noqa: E731
            flip = torch.where(u() < a["flip_prob"], -1.0, 1.0)
            fmat = torch.diag_embed(torch.stack([flip, torch.ones_like(flip), torch.ones_like(flip)], -1))
            r = math.radians(a["rotate_deg"])
            do_rot = (u() < a["rotate_prob"]).float()
            rmat = eye
            for ax in range(3):
                rmat = rmat @ _rot(ax, (torch.rand(b, device=device) * 2 - 1) * r * do_rot)
            lo, hi = a["scale_range"]
            s = torch.where(u() < a["scale_prob"], lo + (hi - lo) * u(), torch.ones(b, device=device))
            m = (rmat @ fmat) / s[:, None, None]  # output mm -> input mm
        sp = torch.tensor(self.spacing, device=device)
        d_out = torch.diag(torch.tensor(self.patch, device=device) * sp / 2)
        d_in_inv = torch.diag(2.0 / (torch.tensor(self.big, device=device) * sp))
        theta = d_in_inv @ m @ d_out                       # normalised coords, (x, y, z) order
        theta = theta.flip(-1).flip(-2)                    # grid_sample expects (z, y, x) = (W, H, D)
        return torch.cat([theta, torch.zeros(b, 3, 1, device=device)], -1)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, label: torch.Tensor, dist: torch.Tensor | None = None, train: bool = True):
        b = image.shape[0]
        theta = self._theta(b, image.device, train)
        grid = F.affine_grid(theta, [b, 1, *self.patch], align_corners=False)
        image = F.grid_sample(image.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
        label = F.grid_sample(label.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
        if dist is not None:
            dist = F.grid_sample(dist.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
        if train:
            image = self._intensity(image)
        return image, label, dist

    def _intensity(self, x: torch.Tensor) -> torch.Tensor:
        a, b, dev = self.a, x.shape[0], x.device
        p = a["intensity_prob"]
        shape = (b, 1, 1, 1, 1)
        on = lambda: (torch.rand(shape, device=dev) < p).float()  # noqa: E731
        shift = (torch.rand(shape, device=dev) * 2 - 1) * a["intensity_shift"] * on()
        lo, hi = a["contrast_range"]
        fac = 1.0 + (lo + (hi - lo) * torch.rand(shape, device=dev) - 1.0) * on()
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        x = (x - mean) * fac + mean + shift
        std = torch.rand(shape, device=dev) * a["noise_std"] * on()
        return x + torch.randn_like(x) * std
