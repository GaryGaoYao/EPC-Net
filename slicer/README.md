# EPCNet — 3D Slicer extension

Automatic, anatomy-guided segmentation of the **orbital bone** (and the orbital content) in head CT / CBCT with
[EPC-Net](https://github.com/GaryGaoYao/EPC-Net). The output segmentation can be reviewed and edited in Segment Editor, then exported as STL for patient-specific orbital implant design.

![EPCNet module in 3D Slicer](Screenshot01.png)

## Module

**EPC-Net Orbital Bone** (category *Segmentation*) runs a two-stage cascade:
- Stage I localises the orbital content.
- Its distance-based priors guide Stage II, which segments the thin orbital walls at 0.5 × 0.5 × 1.0 mm.
- Results are mapped back to the native resolution of the input volume.

## Installation

1. Install **EPCNet** from the Extensions Manager. This also installs the required *PyTorch* extension. Restart Slicer.
2. On first use, the module asks to install PyTorch, MONAI and the `epcnet` Python package into Slicer's Python environment. This needs a network connection and can take several minutes, because PyTorch with GPU support is about 2–3 GB.

## Model weights

The trained weights are **not** distributed with the extension. The EPC-Net v1.0.0 public-data weights are released on request after approval by the corresponding authors; see [`weights/`](https://github.com/GaryGaoYao/EPC-Net/tree/main/weights) for the procedure.
Select `stage1_eye.pth` and `stage2_bone.pth` in the *Model weights* section. The paths are remembered.

## Usage

1. Load a head CT/CBCT volume (DICOM or NIfTI).
2. Open *Segmentation → EPC-Net Orbital Bone*. Select the input volume and the two weight files, then click **Apply**.
3. Review and correct the *Orbital bone* segment in Segment Editor.
4. Click **Export orbital bone surface (STL)…**. The export uses the current segment, including your manual edits. The surface is built as in the paper: Gaussian σ = 1 voxel, Marching Cubes at 0.5, and 20 Laplacian iterations. Coordinates are LPS.

Options:
- **Region**:
  - *Orbital region (fast)*, the default, runs Stage II only around the orbits found by Stage I. Inside that region the result is identical to *whole volume*, and it is several times faster, which matters most without a GPU.
  - *Whole volume* reproduces the paper's setting.
- **Device**: Auto / CUDA GPU / CPU.
- **Window overlap**: 0.5 as in the paper; 0.25 is about 2× faster.

Typical run time per case is under 1–2 minutes on an NVIDIA GPU. On CPU it takes a few minutes in the orbital-region mode.

## Disclaimer

Research use only. This is not a medical device. All segmentations and surfaces must be reviewed by qualified users before any clinical or manufacturing use.

## Citation

Gao Y. et al. *Fast and anatomy-guided orbital bone segmentation to support Patient-Specific Implant design.*
Biomedical Signal Processing and Control 130 (2027) 111465. https://doi.org/10.1016/j.bspc.2026.111465

## Developer notes

- **Try it without building.** Go to *Edit → Application Settings → Modules → Additional module paths* and add `<EPC-Net checkout>/slicer/SlicerEPCNet/EPCNet`, then restart Slicer. In this mode the module imports `epcnet` directly from the checkout.
- **Extensions Index.** Submit `SlicerEPCNet/EPCNet.json` (with `build_subdirectory: slicer/SlicerEPCNet`) as a pull request to https://github.com/Slicer/ExtensionsIndex.
- **Installed version.** The released module installs `epcnet` from the tag in `EPCNET_PACKAGE_URL` (`EPCNet.py`). Create that tag before releasing.
