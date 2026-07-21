# Orbital Bone Segmentation for PSI Workflows

**EPC-Net Orbital Bone STL Generator** is a lightweight Windows desktop application for automated **orbital bone segmentation from CT and CBCT images**. It is developed to support research in **virtual surgical planning**, **patient-specific implant (PSI) design**, and **medical 3D modelling**.

The application uses a two-stage segmentation pipeline:

1. **Eye globe segmentation** for automatic orbital localization.
2. **Eye-guided orbital bone segmentation** for improved reconstruction of thin and anatomically complex orbital walls.

The predicted orbital bone mask is automatically converted into a smoothed three-dimensional STL model through a simple one-click workflow.

## Application Interface

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/8fad3e9b-485d-4a82-af61-74d3fe502273"
    alt="EPC-Net Orbital Bone STL Generator"
    width="800"
  />
</p>

The interface is designed for users without programming experience. <br>
Users only need to select an input image, specify the STL output path, and start the automated processing pipeline.

## Features
- Windows 11 desktop application
- Import CT or CBCT images in NIfTI format (`.nii` and `.nii.gz`)
- One-click eye globe and orbital bone segmentation
- Eye-guided localization of the orbital region
- Automatic surface extraction from the predicted orbital bone mask
- Automatic Gaussian smoothing and mesh refinement
- Direct export of the reconstructed orbital bone as an STL file
- User-defined STL output location
- GPU-accelerated inference when a compatible CUDA device is available
- Automatic CPU fallback when GPU acceleration is unavailable
- Configurable model paths through the **Model Settings** panel
- No command-line operation or manual Python configuration required

## Input

The current application accepts CT or CBCT volumes in the following formats:

```text
.nii
.nii.gz
```

The input image should contain the complete orbital region and should preserve the original spatial information of the scan.

## Output

The application generates a reconstructed orbital bone surface model in STL format:

```text
orbital_bone.stl
```

The exact file name and storage location can be selected by the user before processing.

The generated STL model can be opened in commonly used medical image-processing and 3D-modelling software, including:

- 3D Slicer
- MeshLab
- Blender
- Materialise 3-matic
- Other CAD or medical 3D-processing platforms

## Installation

Download the latest packaged Windows version from the **Releases** section of this repository.

The application is distributed as a Windows executable or installer and does not require users to manually configure Python or install the required software dependencies.

### System Requirements

- Windows 11
- At least 8 GB RAM recommended
- NVIDIA GPU with CUDA support recommended for faster inference
- CPU-only execution is supported when a compatible GPU is unavailable
- Sufficient disk space for the application, model weights, input images, and generated STL files

## Model Weights

The trained EPC-Net model weights are not included in the installer or source-code repository.

Users must submit an access request and sign the corresponding research-use statement before receiving access to the model weights.

**Model access:** To be updated

After approval, download the model package and configure the model locations through the **Model Settings** panel in the application.

Detailed installation and configuration instructions will be provided together with the approved model package.

## Usage

1. Install and launch **EPC-Net Orbital Bone STL Generator**.
2. Click **Browse** and select a CT or CBCT image in `.nii` or `.nii.gz` format.
3. Click **Save As** and specify the output STL file path.
4. Confirm that the required model weights are correctly configured.
5. Click **Generate STL**.
6. Wait for segmentation, surface extraction, smoothing, and STL export to complete.
7. Review the generated STL model using appropriate medical imaging or 3D-processing software.

## Model Settings

The **Model Settings** panel allows users to configure the paths of the required segmentation models.

Before running the application for the first time, make sure that:

- the eye globe segmentation model is available;
- the orbital bone segmentation model is available;
- the selected files correspond to the approved model package;
- the application has permission to access the selected model files.

The application displays **Models ready** when the required model files have been successfully detected.

## Demo Video

A demonstration video showing installation, model configuration, and STL generation will be released together with the software.

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

## Responsible Use

> [!WARNING]
> This software is intended for research, education, and technical evaluation only. It is not a certified medical device and must not be used as an autonomous system for clinical diagnosis, treatment planning, implant manufacturing, surgical decision-making, or direct patient care.

All generated segmentations and STL models must be carefully reviewed and validated by qualified clinicians, biomedical engineers, or other appropriately trained professionals before further use.

Users are responsible for ensuring that all medical images are handled in accordance with applicable ethical, institutional, privacy, and data-protection requirements.

## License

The software, source code, and trained model weights may be subject to different access and licensing conditions.

The model weights are provided only to approved users under the accompanying research-use statement. Redistribution, commercial use, clinical use, or integration into another product is not permitted unless separately authorized in writing.

## Contact

For software questions, model-access requests, or research collaboration, please contact:

**Yao Gao**  
Department of Oral and Maxillofacial Surgery  
KU Leuven / University Hospitals Leuven  
Leuven, Belgium  

Email: `yao.gao@kuleuven.be`

## Acknowledgements

This project was developed through collaboration among clinicians, biomedical engineers, and computer scientists working on orbital reconstruction, medical image segmentation, patient-specific implant design, and surgical 3D technology.
