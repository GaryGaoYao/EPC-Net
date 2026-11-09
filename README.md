# EPC-Net — orbital bone segmentation

Code for **EPC-Net**, a two-stage network for automatic orbital bone segmentation in head CT / CBCT.

> Y. Gao, P. D. Gómez, X. Du, F. Li, L. Tian, J. Van Dessel, R. Willaert, Y. Sun.
> *Fast and anatomy-guided orbital bone segmentation to support Patient-Specific Implant design.*
> Biomedical Signal Processing and Control 130 (2027) 111465. https://doi.org/10.1016/j.bspc.2026.111465

Stage I finds the orbital content. Its result guides Stage II, which segments the orbital bone.
Method details, hyper-parameters and experiments are described in the paper.

## Installation

```bash
git clone https://github.com/GaryGaoYao/EPC-Net.git
cd EPC-Net
pip install torch --index-url https://download.pytorch.org/whl/cu121   # PyTorch for your CUDA version
pip install -e .
```

## Prediction

1. Request the model weights by e-mail (see [Model weights](#model-weights)).
2. Put `stage1_eye.pth` and `stage2_bone.pth` in `weights/`.
3. Run:

```bash
python -m epcnet predict -i CT.nii.gz -d output/ --stl
```

The input can be a `.nii` / `.nii.gz` file, a folder of NIfTI files, or a DICOM series folder.

For each case, `output/` contains:

| File | Content |
|---|---|
| `<case>_bone.nii.gz` | orbital bone mask |
| `<case>_orbit.nii.gz` | orbital content mask |
| `<case>.nii.gz` | combined label map (1 = bone, 2 = orbital content) |
| `<case>_bone.stl` | bone surface (with `--stl`) |

All masks are on the original image grid.

You can also run EPC-Net from the desktop GUI in [`gui/`](gui), or from 3D Slicer with the extension (to be updated...).

## Training on your own data

**1. Prepare the data** in nnU-Net raw format: one label map per case containing both the orbital bone and the orbital content.

```
DatasetXXX_Name/
├── dataset.json              # e.g. "labels": {"background": 0, "bone": 1, "eyes": 2}
├── imagesTr/CASE_0000.nii.gz
└── labelsTr/CASE.nii.gz
```

**2. Write a config file** with your paths, e.g. `configs/my_data.yaml`:

```yaml
runs_dir: D:/epcnet_work/runs
data:
  raw_dir: D:/nnUNet_raw/DatasetXXX_Name
  cache_dir: D:/epcnet_work/cache
  split_file: D:/epcnet_work/splits/split.json
  bone_label: bone        # label name in dataset.json
  orbit_label: eyes
```

**3. Train.**

```bash
python -m epcnet preprocess -c configs/my_data.yaml
python -m epcnet split      -c configs/my_data.yaml --n-val 5 --n-test 5
python -m epcnet train      -c configs/my_data.yaml --stage 1
python -m epcnet priors     -c configs/my_data.yaml
python -m epcnet train      -c configs/my_data.yaml --stage 2
python -m epcnet evaluate   -c configs/my_data.yaml --split test
python -m epcnet export     -c configs/my_data.yaml
```

On Windows, all of these steps run in one go with `.\scripts\train_epcnet.ps1 -Config configs/my_data.yaml`.

* The default settings are those of the paper.
* An interrupted training continues with `--resume`.
* `export` writes `stage1_eye.pth` and `stage2_bone.pth` to `release/`.

## Model weights

The weights are trained on public data and are not included in this repository. To get them:

1. Fill in and sign the one-page [request form](weights/Model%20Weight%20Access%20Request%20and%20Research-Use%20Statement.pdf).
2. E-mail it to the corresponding authors with the subject `EPC-Net weight request – <your name>, <institution>`:
   - Jeroen Van Dessel — jeroen.vandessel@kuleuven.be
   - Robin Willaert — robin.willaert@icloud.com
   - Yi Sun — yi.sun@uzleuven.be
3. After the authors approve your request, you will receive a download link.

The weights are for non-commercial research only and may not be shared.
The original weights used in the paper were trained on hospital data; whether they can be shared depends on the decision of the hospital and its Ethics Committee.

## Disclaimer

For research use only. EPC-Net is not a certified medical device.
Every result must be checked by a qualified clinician or engineer before clinical or manufacturing use.

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
