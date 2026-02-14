# EPC-Net GUI — Orbital Bone Segmentation for PSI Workflows

EPC-Net GUI is a desktop application for **orbital bone segmentation on CBCT/CT**, built for **virtual surgical planning** and **patient-specific implant (PSI)** reconstruction.  
It follows a **two-stage cascade**: (1) **eye (globe) segmentation** to localize the orbit, then (2) **eye-guided orbital bone segmentation** for improved thin-wall completeness.

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
- Auto-saves run logs / metadata for traceability

---

## Demo


---

## Quick Start (recommended: binary)
1. Open **Releases** on this repo and download the latest package (e.g., `EPCNet-GUI-win64.zip`).
2. Unzip and run `EPCNetGUI.exe`.
3. Put model weights into `./weights/` (see below).

> If Windows SmartScreen blocks it: **More info → Run anyway**.

---

## Citation

If you use our segmentation tool in your research, please cite our paper (under review somewhere):

### BibTeX
```bibtex
@article{gao2026epcnet_paper,
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


