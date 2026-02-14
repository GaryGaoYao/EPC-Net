# EPC-Net GUI — Orbital Bone Segmentation for PSI Workflows

EPC-Net GUI is a desktop application for **orbital bone segmentation on CBCT/CT**, built for **virtual surgical planning** and **patient-specific implant (PSI)** reconstruction.  It follows a **two-stage cascade**: (1) **eye (globe) segmentation** to localize the orbit, then (2) **eye-guided orbital bone segmentation** for improved thin-wall completeness.
<p align="center">
<img src="https://github.com/user-attachments/assets/76a55bd0-e732-4ac8-8128-e797072595db" alt="Study-Design" width="500">
</p>
---

## Features
- Import **DICOM CT folder** or **NIfTI** (`.nii/.nii.gz`)
- One-click inference:
  - **Stage I:** Eye mask
  - **Stage II:** Orbital bone mask (eye-guided)
- Slice viewer with mask overlay (opacity adjustable)
- Export:
  - Masks: `pred_eye.nii.gz`, `pred_bone.nii.gz` (intermediate products)
  - Mesh: `bone_surface.stl` (final output)

<p align="center">
<img src="https://github.com/user-attachments/assets/288601a0-2ebb-4e8e-84c7-786d63a6f798" alt="Study-Design" width="600">
</p>

---
## Demo Video

To be published~

---

## Quick Start

1. **Read and accept the Software Use Notice (required).**  
   By using this software, you acknowledge that it is provided **for research/educational use only**, **not for clinical or medical use**, and that the authors/institutions assume **no liability** for any outcomes resulting from its use.

2. Download the latest package of our software from Zenodo (link to be updated): https://zenodo.org/

3. Download the trained model weights and place them under `./weights/` (or select them in the GUI if supported).

4. Run the application and perform inference. The predicted masks and optional STL mesh will be saved to the default output directory (or a custom save path if specified).  
   You can find the exported STL mesh file (e.g., `bone_surface.stl`) in the output folder.

---

## Citation

If you use our segmentation tool in your research, please cite our paper (under review somewhere):

### BibTeX
```bibtex
@unpublished{gao2026epcnet_paper,
  title   = {Fast and Topology-Robust Orbital Bone Segmentation to Support Patient-Specific Implant Design},
  author  = {Gao, Yao and others},
  journal = {<JOURNAL>},
  year    = {2026},
  volume  = {<VOLUME>},
  number  = {<NUMBER>},
  pages   = {<PAGES>},
  doi     = {<DOI>}
}
```
### APA (7th edition)
```apatex
Gao, Y., Gómez, P. D., Du, X., Li, F., Tian, L., Van Dessel, J., Willaert, R., & Sun, Y. (2026). Fast and topology-robust orbital bone segmentation to support patient-specific implant design [Manuscript under review].
```
