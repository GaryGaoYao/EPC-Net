# Three-Stage Orbital Segmentation and Mesh Reconstruction

This repository contains the three core scripts used for a cascaded orbital image-processing workflow:

1. **Stage 1 — Eye-region localization:** train a 3D SegResNet on CT volumes and binary eye-region labels.
2. **Stage 2 — Eye-guided bone segmentation:** train a second 3D SegResNet using the CT image and an eye-distance guidance map, with an eye-weighted Dice and cross-entropy loss.
3. **Stage 3 — Mesh reconstruction:** convert a selected label in a NIfTI segmentation into an STL mesh using Marching Cubes and optional windowed-sinc smoothing.

The code is organized for public research release. It removes machine-specific paths, matches cases by identifiers rather than list order, records the exact train/validation split, saves complete resumable checkpoints, and supports a reproducible command-line workflow.

## Repository structure

```text
.
├── stage1_train_eye_locator.py
├── stage2_train_eye_guided_bone.py
├── stage3_reconstruct_mesh.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

Install a PyTorch build appropriate for the local CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The PyQt5 dependency is only required for the optional Stage 3 graphical interface. Stage 3 command-line reconstruction does not import PyQt5.

## Data organization

### Stage 1

```text
Dataset_EyeLocator/
├── imagesTr/
│   ├── case001_0000.nii.gz
│   ├── case002_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── case001.nii.gz
    ├── case002.nii.gz
    └── ...
```

Image and label files are matched by `caseXXX`. The script stops with an explicit error when any case is missing its counterpart.

### Stage 2

```text
Dataset_BoneSeg/
├── imagesTr/
│   ├── case001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case001.nii.gz
│   └── ...
└── eyePredTr/
    ├── case001_0000_pred_origspace.nii.gz
    └── ...
```

The eye predictions must be spatially aligned with the corresponding CT and label volumes. A different eye-prediction directory can be supplied with `--eye-dir`.

## Usage

### Stage 1: eye-region locator training

```bash
python stage1_train_eye_locator.py \
  --data-dir /path/to/Dataset_EyeLocator \
  --output-dir outputs/stage1 \
  --gpu 0
```

Important defaults reproduce the supplied configuration:

- isotropic spacing: `1.0 1.0 1.0` mm;
- patch size: `96 96 96`;
- batch size: `8`;
- maximum epochs: `500`;
- validation interval: every `5` epochs;
- SegResNet input/output channels: `1/2`.

Resume an interrupted run from its complete checkpoint:

```bash
python stage1_train_eye_locator.py \
  --data-dir /path/to/Dataset_EyeLocator \
  --resume outputs/stage1/run_YYYYMMDD_HHMMSS/latest_checkpoint.pth
```

### Stage 2: eye-guided bone segmentation training

```bash
python stage2_train_eye_guided_bone.py \
  --data-dir /path/to/Dataset_BoneSeg \
  --output-dir outputs/stage2 \
  --gpu 0
```

Important defaults reproduce the supplied configuration:

- target spacing: `0.5 0.5 1.0` mm;
- patch size: `128 128 128`;
- two input channels: CT and eye-distance guidance;
- `sigma = 5.0` mm and `alpha = 6.0` for eye weighting;
- sampling band: `0–25` mm from the dilated eye mask;
- warm-up: `30` epochs;
- ReduceLROnPlateau factor: `0.5`;
- full-checkpoint resume, AMP-safe gradient clipping, and CSV logging.

Resume an interrupted run:

```bash
python stage2_train_eye_guided_bone.py \
  --data-dir /path/to/Dataset_BoneSeg \
  --resume outputs/stage2/run_YYYYMMDD_HHMMSS/latest_checkpoint.pth
```

### Stage 3: NIfTI label to STL

Command-line reconstruction:

```bash
python stage3_reconstruct_mesh.py \
  --input /path/to/segmentation.nii.gz \
  --output /path/to/reconstruction.stl \
  --label 1 \
  --smoothing-iterations 30 \
  --passband 0.02
```

Optional GUI:

```bash
python stage3_reconstruct_mesh.py --gui
```

The requested integer label is converted to a binary mask before Marching Cubes. This avoids unintentionally merging higher-valued labels in multiclass segmentations. Image spacing, origin, and direction are applied to the exported physical-space mesh.

## Training outputs

Each Stage 1 or Stage 2 run contains:

```text
run_YYYYMMDD_HHMMSS/
├── best_checkpoint.pth
├── best_model_weights.pth
├── latest_checkpoint.pth
├── latest_model_weights.pth
├── run_manifest.json
└── training_log.csv
```

`run_manifest.json` records the arguments, software versions, device, and exact training/validation case identifiers. A full checkpoint contains model, optimizer, gradient-scaler, scheduler where applicable, epoch, best metric, and random-number-generator states.

## Reproducibility notes

- The train/validation split is shuffled with a fixed seed and recorded in the run manifest.
- Stage 2 physical-distance maps use the configured anisotropic spacing explicitly.
- Automatic mixed precision is enabled only on CUDA and can be disabled with `--no-amp`.
- The code does not include patient data, pretrained weights, or institutional file paths.
- Exact numerical reproducibility can still depend on the operating system, GPU, CUDA, PyTorch, and MONAI versions.

## Responsible use

This research code is not a certified medical device. It must not be used for autonomous diagnosis, treatment planning, implant manufacturing, or other clinical decisions without appropriate validation and qualified expert oversight.

## Citation

Please cite the associated paper when using this code. Replace this section with the final bibliographic record or BibTeX entry after publication.

## License

No software license is selected in this draft package. The repository owner should add the intended open-source license before public release.
