# EPC-Net — anatomy-guided orbital bone segmentation

Reference implementation of **EPC-Net (Eye-Prior Cascade Net)** from

> Y. Gao, P. D. Gómez, X. Du, F. Li, L. Tian, J. Van Dessel, R. Willaert, Y. Sun.
> *Fast and anatomy-guided orbital bone segmentation to support Patient-Specific Implant design.*
> Biomedical Signal Processing and Control 130 (2027) 111465. https://doi.org/10.1016/j.bspc.2026.111465

EPC-Net is a two-stage cascade. **Stage I** segments the orbital content with a 3D SegResNet at 1 mm.
The prediction is dilated (δ = 1.5 mm) and converted to a distance field *d*, which gives two priors.
The **feature prior** `G = exp(−d/σ)` is concatenated to the CT as a second input channel.
The **optimisation prior** `W = 1 + α·exp(−d²/2σ²)` weights both terms of the Dice + cross-entropy loss.
**Stage II** then segments the orbital bone at 0.5 × 0.5 × 1.0 mm, with patch centres drawn preferentially from the peri-orbital band (d ≤ d_max).

| | Stage I (orbital content) | Stage II (orbital bone) |
|---|---|---|
| Backbone | MONAI SegResNet 3D, 32 filters, GroupNorm, dropout 0.2 | same |
| Spacing | 1.0 × 1.0 × 1.0 mm | 0.5 × 0.5 × 1.0 mm |
| Patch | 128 × 128 × 128 | 160 × 160 × 96 |
| Input | CT | CT + G |
| Loss | Dice + CE | weighted Dice + weighted CE (W) |
| Sampling | random | peri-orbital focused |
| Optimiser | AdamW 1e-4, wd 1e-5, cosine, batch 2, ≤500 epochs, early stop 50 | same |
| Priors | – | σ = 5 mm, α = 6, δ = 1.5 mm, d_max = 25 mm |

All settings are in [`epcnet/default_config.yaml`](epcnet/default_config.yaml).

**Resolution.** The networks run on the grids of Table 2.
The Stage-II bone probability map is then interpolated back to the native CT grid before thresholding at 0.5, so masks and STL surfaces keep the full scanner resolution (for example 0.29 × 0.29 × 0.3 mm).
`configs/highres_isotropic.yaml` is an optional extension that runs Stage II at 0.5 mm isotropic; it is *not* the published setting.

**GPU throughput.** The DataLoader only cuts crops from the memory-mapped cache.
Flip, rotation and scaling (done in mm space) and the intensity augmentation all run batch-wise on the GPU.
Training uses AMP, cuDNN benchmark and TF32, and validation runs 8 sliding windows per forward pass.

## Installation

```bash
# install the matching CUDA build of PyTorch first, then
pip install -e .
python tests/smoke_test.py      # ~3 min CPU check of the whole pipeline on synthetic data
```

## Data

The training data uses the **nnU-Net raw format**, so an existing nnU-Net dataset can be used without changes:

```
DatasetXXX_Name/
├── dataset.json          # labels, e.g. {"background": 0, "bone": 1, "eyes": 2}
├── imagesTr/CASE_0000.nii.gz
└── labelsTr/CASE.nii.gz  # one label map holding orbital bone AND orbital content
```

The label names (or integer values) are set with `data.bone_label` and `data.orbit_label`.

## Training

```bash
CFG=configs/local_dataset777.yaml           # paths to your data / cache / runs
python -m epcnet preprocess -c $CFG          # resample + normalise into the cache
python -m epcnet split      -c $CFG --n-val 5 --n-test 5
python -m epcnet train      -c $CFG --stage 1
python -m epcnet priors     -c $CFG          # Stage-I predictions -> distance priors
python -m epcnet train      -c $CFG --stage 2
python -m epcnet evaluate   -c $CFG --split test --save-predictions
python -m epcnet export     -c $CFG          # weights-only release package
```

On Windows the whole sequence is in `scripts/train_epcnet.ps1`.
The 2 × 2 ablation (Table 7) is in `scripts/run_ablation.ps1`.

* Any key can be overridden with `-o`, for example `-o prior.sigma=3.5 -o experiment=sigma3.5`. Distances are cached once, so σ, α, δ and d_max can be changed without preprocessing again (OAT sensitivity analysis, Sec. 3.15).
* Interrupted training continues with `train --stage N --resume`.
* `split` also writes `splits_final_for_nnunet.json`. Copy it to `nnUNet_preprocessed/<dataset>/splits_final.json` to train an nnU-Net baseline on the same patients (fold 0).
* `evaluate` reports global and PO-ROI (1.1–1.9×) DSC / HD95 / ASD at the original resolution. It also reports Stage-I DSC, HD95 and centroid error, and the end-to-end time per case.

## Inference

```bash
python -m epcnet predict -i case_0000.nii.gz -d out/ --stl
python -m epcnet predict -i folder_with_niftis/ -d out/
python -m epcnet predict -i dicom_series_folder/ -d out/ --stl
python -m epcnet predict -i case_0000.nii.gz -d out/ --roi-margin 40   # fast: Stage II only around the orbits
```

Outputs are written on the original image grid: `<case>_bone.nii.gz`, `<case>_orbit.nii.gz`, and `<case>.nii.gz` (1 = bone, 2 = orbital content).
With `--stl`, the bone surface is also saved as `<case>_bone.stl`, built as in Sec. 3.9: Gaussian σ = 1 voxel, Marching Cubes at 0.5, and 20 Laplacian iterations with relaxation 0.01, in LPS coordinates.

```python
from epcnet.inference import EPCNetPredictor
predictor = EPCNetPredictor("weights/stage1_eye.pth", "weights/stage2_bone.pth")
result = predictor.predict("case_0000.nii.gz")   # result["bone"], result["orbit"], result["timings"]
```

The exported weights (`stage1_eye.pth`, `stage2_bone.pth`) are plain SegResNet state dicts with metadata.

## 3D Slicer extension

The **EPCNet** extension for [3D Slicer](https://www.slicer.org) lives in [`slicer/`](slicer). It segments the orbital bone of the loaded volume into an editable segmentation node and exports the reviewed bone surface as STL. Users select their own weight files.

## Desktop application

The **EPC-Net Orbital Bone STL Generator** (Windows, PyQt5) is in [`gui/`](gui), in English and Chinese, and installers are on the [Releases](https://github.com/GaryGaoYao/EPC-Net/releases/latest) page.
It loads `weights/stage1_eye.pth` and `weights/stage2_bone.pth` automatically.
It runs the same `epcnet` inference engine as the CLI, reading spacing and patch size from the weight metadata, and writes the surface as in Sec. 3.9.

## Model weights

Trained weights are **not** included in this repository. There are two sets:

* **EPC-Net v1.0.0 public-data weights**: trained on publicly available CT data only, with this repository's code and the paper's Table 2 settings. To request them, e-mail the signed access form to the corresponding authors. They are released only after the authors approve the request.
* **Original weights reported in the paper**: trained on the University Hospitals Leuven clinical cohort. Whether these can be shared is decided by the hospital and its Ethics Committee.

The procedure and contact addresses are in [`weights/README.md`](weights/README.md).

## Responsible use

This software is intended for research only. It is not a certified medical device.
Every segmentation and surface must be reviewed by qualified clinicians or engineers before any clinical or manufacturing use.

## Citation

```bibtex
@article{gao2027epcnet,
  title   = {Fast and anatomy-guided orbital bone segmentation to support Patient-Specific Implant design},
  author  = {Gao, Yao and G{\'o}mez, Pedro Dami{\'a}n and Du, Xijin and Li, Feng and Tian, Lei and
             Van Dessel, Jeroen and Willaert, Robin and Sun, Yi},
  journal = {Biomedical Signal Processing and Control},
  volume  = {130},
  pages   = {111465},
  year    = {2027},
  doi     = {10.1016/j.bspc.2026.111465}
}
```
