# EPC-Net model weights

No trained weights are included in this repository. There are two sets of weights, and they are shared under different conditions.

| | Public-data weights (EPC-Net v1.0.0) | Original weights reported in the paper |
|---|---|---|
| Training data | Publicly available CT data only | In-house clinical cohort, University Hospitals Leuven |
| Training code | This repository (v1.0.0, paper settings of Table 2) | Original study code |
| Availability | **On request, after approval by the corresponding authors** | **Subject to the decision of the hospital / Ethics Committee** |

## 1. Public-data weights (EPC-Net v1.0.0) — request procedure

These weights were trained exclusively on publicly available data, using the code and configuration in this repository.
The package contains `stage1_eye.pth`, `stage2_bone.pth`, `epcnet_config.yaml` and a `manifest.json` with SHA-256 checksums.

1. Complete and sign the [`Model Weight Access Request and Research-Use Statement.pdf`](./Model%20Weight%20Access%20Request%20and%20Research-Use%20Statement.pdf).
2. E-mail the signed form to the **corresponding authors**, with the repository maintainer in copy:
   * Jeroen Van Dessel — jeroen.vandessel@kuleuven.be
   * Robin Willaert — robin.willaert@icloud.com
   * Yi Sun — yi.sun@uzleuven.be
   * cc: Yao Gao — yao.gao@kuleuven.be

   Use the subject line `EPC-Net weight access request – <name>, <institution>`. Briefly describe the intended research use.
3. The corresponding authors review each request. Access is granted only after their written approval and is at their discretion.
   Approved applicants receive a personal download link.

Conditions of use, as stated in the request form:

* research use only — not for clinical decision-making, and not a certified medical device;
* no redistribution of the weights;
* cite the EPC-Net paper (doi:10.1016/j.bspc.2026.111465).

Users must also comply with the licence and usage terms of the public datasets the model was trained on.

## 2. Original weights reported in the paper

The models evaluated in the paper were trained on retrospectively collected clinical data from University Hospitals Leuven / KU Leuven (Ethics Committee approval No. S68944).
Whether, and under which conditions, these weights can be shared is determined by the hospital and its Ethics Committee.
They are **not** released through the procedure above.
Enquiries can be sent to the corresponding authors, who will forward them for institutional and ethical review.

## Installing the weights

After approval, place the two files in this folder:

```text
./weights/
├── stage1_eye.pth
└── stage2_bone.pth
```

The CLI (`python -m epcnet predict`), the desktop GUI and the 3D Slicer extension can all load them.
Check the files against the SHA-256 checksums in `manifest.json`.
