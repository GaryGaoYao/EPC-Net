# Eye-Guided Orbital Bone Segmentation for PSI Workflows

A Windows desktop application for automated **orbital bone segmentation from CT and CBCT images**, developed to support **virtual surgical planning** and **patient-specific implant (PSI) design**.

The application uses a two-stage pipeline:

1. **Eye globe segmentation** for orbital localization.
2. **Eye-guided orbital bone segmentation** for improved reconstruction of thin orbital walls.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76a55bd0-e732-4ac8-8128-e797072595db"
       alt="Orbital bone segmentation workflow"
       width="500">
</p>

## Features

* Windows 11 desktop application
* Import DICOM CT/CBCT folders or NIfTI files (`.nii` and `.nii.gz`)
* One-click eye globe and orbital bone segmentation
* Slice viewer with adjustable mask overlay
* Export segmentation masks and STL meshes

## Outputs

* `pred_eye.nii.gz`: eye globe mask
* `pred_bone.nii.gz`: orbital bone mask
* `bone_surface.stl`: reconstructed orbital bone mesh

## Installation

Download and install the latest Windows version from the **Releases** section of this repository.

The application is provided as a packaged Windows installer and does not require users to configure the Python environment manually.

## Model Weights

The trained model weights are not included in the installer.

Users must submit an access request before downloading the weights from Zenodo:

**Model access:** To be updated

After approval, download the model weights and place them in the following folder:

```text
./weights/
```

Detailed instructions will be provided together with the approved model package.

## Usage

1. Install and launch the application.
2. Select a DICOM folder or NIfTI image.
3. Select the approved model weights.
4. Choose an output directory.
5. Run the automated segmentation.
6. Review and export the masks and STL mesh.

## Demo Video

A demonstration video will be released with the software.

## Citation

Please cite the associated manuscript when using this software or its trained models:

```bibtex
@unpublished{gao2026orbital,
  title  = {Fast and Topology-Robust Orbital Bone Segmentation to Support Patient-Specific Implant Design},
  author = {Gao, Yao and Gómez, Pedro Damián and Du, Xijin and Li, Feng and Tian, Lei and Van Dessel, Jeroen and Willaert, Robin and Sun, Yi},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Disclaimer

> [!WARNING]
> This software is provided for research and educational purposes only. It is not a certified medical device and must not be used for clinical diagnosis, treatment planning, implant manufacturing, or direct patient care. All outputs must be reviewed by qualified experts.
