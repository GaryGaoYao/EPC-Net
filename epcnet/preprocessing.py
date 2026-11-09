"""Image I/O and the resampling / normalisation shared by training and inference.

Every volume is (1) loaded as a channel-first MONAI ``MetaTensor`` that carries its
RAS affine, (2) re-oriented to RAS, (3) resampled to the stage spacing and
(4) scaled to [0, 1] with robust 0.5/99.5 percentiles.  The exact same functions
are used for preprocessing the training cache and for inference, so the two can
never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import LoadImage, Orientation, ResampleToMatch, Spacing, SpatialResample

NIFTI_SUFFIXES = (".nii", ".nii.gz")


# ----------------------------------------------------------------------------- I/O
def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _sitk_to_metatensor(image) -> MetaTensor:
    """SimpleITK image (LPS world) -> channel-first MetaTensor with RAS affine, array order (x, y, z)."""
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    arr = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    affine = np.eye(4)
    affine[:3, :3] = lps_to_ras @ direction @ np.diag(spacing)
    affine[:3, 3] = lps_to_ras @ origin
    return MetaTensor(torch.from_numpy(arr[None].astype(np.float32)), affine=torch.as_tensor(affine))


def load_volume(path: str | Path, dtype=np.float32) -> MetaTensor:
    """Load NIfTI / NRRD / MHA or a DICOM series folder as a channel-first MetaTensor."""
    path = Path(path)
    if path.is_file() and _is_nifti(path):
        img = LoadImage(image_only=True, ensure_channel_first=True, reader="NibabelReader", dtype=dtype)(str(path))
        return img
    import SimpleITK as sitk

    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(path))
        if not series:
            raise FileNotFoundError(f"No DICOM series found in {path}")
        # take the series with the most slices
        files = max((reader.GetGDCMSeriesFileNames(str(path), s) for s in series), key=len)
        reader.SetFileNames(files)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(path))
    return _sitk_to_metatensor(image)


def save_mask_like(mask: np.ndarray, reference_path: str | Path | None, affine: np.ndarray, out_path: str | Path) -> None:
    """Save a uint8 mask; reuse the reference NIfTI header when available."""
    import nibabel as nib

    mask = np.asarray(mask).astype(np.uint8)
    header = None
    if reference_path is not None and Path(reference_path).is_file() and _is_nifti(Path(reference_path)):
        ref = nib.load(str(reference_path))
        if ref.shape[:3] == mask.shape:
            header = ref.header.copy()
            affine = ref.affine
    img = nib.Nifti1Image(mask, np.asarray(affine, dtype=np.float64), header=header)
    img.set_data_dtype(np.uint8)
    img.header["scl_slope"] = 1
    img.header["scl_inter"] = 0
    nib.save(img, str(out_path))


# ------------------------------------------------------------------ geometry
def to_ras(img: MetaTensor, axcodes: str = "RAS") -> MetaTensor:
    return Orientation(axcodes=axcodes, labels=(("L", "R"), ("P", "A"), ("I", "S")))(img)


def resample_to_spacing(img: MetaTensor, spacing: Sequence[float], mode: str = "bilinear") -> MetaTensor:
    return Spacing(pixdim=tuple(float(s) for s in spacing), mode=mode, padding_mode="border" if mode != "nearest" else "zeros",
                   dtype=torch.float32)(img)


def resample_to_grid(img: MetaTensor, dst_affine, dst_shape: Sequence[int], mode: str = "bilinear") -> MetaTensor:
    """Resample a MetaTensor onto another grid given by affine + spatial shape."""
    return SpatialResample(mode=mode, padding_mode="zeros", dtype=torch.float32)(
        img, dst_affine=torch.as_tensor(np.asarray(dst_affine, dtype=np.float64)), spatial_size=tuple(int(s) for s in dst_shape)
    )


def resample_to_reference(img: MetaTensor, reference: MetaTensor, mode: str = "bilinear") -> MetaTensor:
    return ResampleToMatch(mode=mode, padding_mode="zeros", dtype=torch.float32)(img, img_dst=reference)


def spacing_from_affine(affine) -> tuple[float, float, float]:
    a = np.asarray(affine, dtype=np.float64)
    return tuple(float(v) for v in np.sqrt((a[:3, :3] ** 2).sum(axis=0)))


# ---------------------------------------------------------------- intensity
def percentile_normalize(img: MetaTensor, percentiles: Sequence[float] = (0.5, 99.5)) -> tuple[MetaTensor, tuple[float, float]]:
    arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    lo, hi = np.percentile(arr, list(percentiles))  # whole volume, as MONAI ScaleIntensityRangePercentiles
    lo, hi = float(lo), float(hi)
    scale = (hi - lo) if hi > lo else 1.0
    out = torch.clamp((img - lo) / scale, 0.0, 1.0)
    return out, (lo, hi)


def prepare_stage_input(img_ras: MetaTensor, spacing: Sequence[float], percentiles: Sequence[float]) -> tuple[MetaTensor, dict]:
    """RAS image -> resampled + normalised image for one stage, plus geometry metadata."""
    resampled = resample_to_spacing(img_ras, spacing, mode="bilinear")
    normed, (lo, hi) = percentile_normalize(resampled, percentiles)
    meta = {
        "affine": np.asarray(resampled.affine, dtype=np.float64).tolist(),
        "shape": [int(s) for s in resampled.shape[1:]],
        "spacing": [float(s) for s in spacing],
        "intensity_percentile_values": [lo, hi],
    }
    return normed, meta


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def read_meta(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
