"""End-to-end smoke test on a tiny synthetic dataset (CPU, ~2-5 minutes).

Creates fake head CTs with two 'eyes' (orbital content) surrounded by thin 'bone' shells,
then runs: preprocess -> split -> train stage 1 -> priors -> train stage 2 -> evaluate ->
export -> predict (+STL).  It checks that the whole pipeline runs, not that it is accurate.

    python tests/smoke_test.py [--keep]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epcnet.cli import main  # noqa: E402


def make_case(path_img: Path, path_lbl: Path, rng: np.random.Generator, shape=(72, 64, 40), spacing=(0.8, 0.8, 1.2)):
    x, y, z = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing="ij")
    ct = rng.normal(-700, 30, shape).astype(np.float32)
    lbl = np.zeros(shape, np.uint8)
    cz = shape[2] * spacing[2] / 2 + rng.uniform(-3, 3)
    for cx in (shape[0] * spacing[0] * 0.3, shape[0] * spacing[0] * 0.7):
        cx += rng.uniform(-2, 2)
        cy = shape[1] * spacing[1] * 0.45 + rng.uniform(-2, 2)
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        eye, shell = r < 8.0, (r >= 10.0) & (r < 11.5) & (y > cy - 4)
        ct[eye] = rng.normal(20, 10, eye.sum())
        ct[shell] = rng.normal(900, 80, shell.sum())
        lbl[shell], lbl[eye] = 1, 2
    aff = np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])  # LPS-like storage to exercise reorientation
    nib.save(nib.Nifti1Image(ct.astype(np.int16), aff), str(path_img))
    nib.save(nib.Nifti1Image(lbl, aff), str(path_lbl))


def run(tmp: Path) -> None:
    raw = tmp / "nnUNet_raw" / "Dataset999_Synthetic"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(8):
        make_case(raw / "imagesTr" / f"case{i:02d}_0000.nii.gz", raw / "labelsTr" / f"case{i:02d}.nii.gz", rng)
    (raw / "dataset.json").write_text(json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "bone": 1,
                                       "eyes": 2}, "numTraining": 8, "file_ending": ".nii.gz"}))
    o = ["-o", f"data.raw_dir={raw.as_posix()}", "-o", f"data.cache_dir={(tmp / 'cache').as_posix()}",
         "-o", f"data.split_file={(tmp / 'splits' / 'split.json').as_posix()}", "-o", f"runs_dir={(tmp / 'runs').as_posix()}",
         "-o", "data.num_workers=0", "-o", "data.preprocess_workers=1",
         "-o", "stage1.spacing=[2.0,2.0,2.0]", "-o", "stage1.patch_size=[32,32,32]",
         "-o", "stage2.spacing=[1.0,1.0,1.5]", "-o", "stage2.patch_size=[48,48,32]",
         "-o", "train.max_epochs=3", "-o", "train.iterations_per_epoch=3", "-o", "train.amp=false",
         "-o", "train.val_device=cpu", "-o", "model.init_filters=8", "-o", "inference.sw_overlap=0.25"]
    main(["preprocess", *o])
    main(["split", "--n-val", "2", "--n-test", "2", *o])
    main(["train", "--stage", "1", *o])
    main(["priors", *o])
    main(["train", "--stage", "2", *o])
    main(["train", "--stage", "2", "--resume", *o, "-o", "train.max_epochs=4"])
    for g, wl in ((False, False), (True, False), (False, True)):  # ablation variants
        main(["train", "--stage", "2", *o, "-o", f"experiment=abl_{g}_{wl}", "-o", f"stage2.use_guidance={g}",
              "-o", f"stage2.use_weighted_loss={wl}", "-o", "train.max_epochs=1"])
    main(["evaluate", *o, "--save-predictions"])
    rel = tmp / "release" / "EPC-Net-weights-vtest"
    main(["export", *o, "--out", str(rel)])
    main(["predict", *o, "-i", str(raw / "imagesTr" / "case00_0000.nii.gz"), "-d", str(tmp / "pred"),
          "--stage1", str(rel / "weights" / "stage1_eye.pth"), "--stage2", str(rel / "weights" / "stage2_bone.pth"), "--stl"])
    # quick integrity checks
    out = nib.load(str(tmp / "pred" / "case00_bone.nii.gz"))
    ref = nib.load(str(raw / "imagesTr" / "case00_0000.nii.gz"))
    assert out.shape == ref.shape and np.allclose(out.affine, ref.affine), "prediction not on the original grid"
    assert (tmp / "pred" / "case00_bone.stl").stat().st_size > 84, "STL missing"
    print("\nSMOKE TEST PASSED ->", tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    d = Path(tempfile.mkdtemp(prefix="epcnet_smoke_"))
    try:
        run(d)
    finally:
        if not a.keep:
            shutil.rmtree(d, ignore_errors=True)
