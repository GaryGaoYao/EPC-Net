"""Command-line interface:  ``python -m epcnet <command> [options]``  (or ``epcnet <command>`` after pip install).

Commands
  preprocess   resample nnU-Net raw data into the EPC-Net cache (both stages)
  split        create the patient-level train/val/test split
  train        train Stage 1 (orbital content) or Stage 2 (orbital bone)
  priors       run the trained Stage-1 model on all cached cases -> Stage-2 distance priors
  evaluate     end-to-end evaluation on the test (or val) split, global + PO-ROI metrics
  predict      segment new CT/CBCT volumes (file, folder of NIfTI files, or DICOM folder)
  export       write a weights-only release package (GUI-compatible file names)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import config as C


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", action="append", default=[],
                   help="YAML overriding epcnet/default_config.yaml; repeatable, later files win")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config key, e.g. -o train.max_epochs=300 (repeatable)")


def _cfg(args):
    return C.load_config(args.config, args.override)


def _ckpt(cfg, stage: int, given: str | None) -> Path:
    p = Path(given) if given else C.run_dir(cfg, stage) / "best_model.pt"
    if not p.exists():
        sys.exit(f"Checkpoint not found: {p}")
    return p


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="epcnet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"EPC-Net {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess", help="build the training cache from nnU-Net raw data")
    _common(p)
    p.add_argument("--cases", nargs="*", default=None)

    p = sub.add_parser("split", help="create train/val/test split")
    _common(p)
    p.add_argument("--n-val", type=int, default=5)
    p.add_argument("--n-test", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--from-nnunet", default=None, help="use train/val of an nnU-Net splits_final.json fold instead")
    p.add_argument("--fold", type=int, default=0)

    p = sub.add_parser("train", help="train one stage")
    _common(p)
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--resume", action="store_true", help="continue from last_checkpoint.pt")

    p = sub.add_parser("priors", help="Stage-1 predictions -> Stage-2 distance priors")
    _common(p)
    p.add_argument("--stage1", default=None, help="Stage-1 checkpoint (default: runs/<experiment>/stage1/best_model.pt)")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("evaluate", help="end-to-end evaluation on raw images")
    _common(p)
    p.add_argument("--stage1", default=None)
    p.add_argument("--stage2", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default=None)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("predict", help="segment new images")
    _common(p)
    p.add_argument("-i", "--input", required=True, help="image file, folder of .nii/.nii.gz files, or DICOM folder")
    p.add_argument("-d", "--output", required=True, help="output folder")
    p.add_argument("--stage1", default="weights/stage1_eye.pth")
    p.add_argument("--stage2", default="weights/stage2_bone.pth")
    p.add_argument("--stl", action="store_true", help="also export the bone surface as STL")
    p.add_argument("--no-orbit", action="store_true", help="do not save the orbital-content mask")
    p.add_argument("--device", default="auto")
    p.add_argument("--overlap", type=float, default=None)
    p.add_argument("--roi-margin", type=float, default=None,
                   help="fast mode: run Stage II only around the orbits (margin in mm, e.g. 40); default = whole volume")

    p = sub.add_parser("export", help="weights-only release package")
    _common(p)
    p.add_argument("--stage1", default=None)
    p.add_argument("--stage2", default=None)
    p.add_argument("--out", default=None, help="default: release/EPC-Net-weights-v<version>")
    p.add_argument("--release-version", default=None)
    p.add_argument("--no-zip", action="store_true")

    args = parser.parse_args(argv)
    from .trainer import setup_logging

    setup_logging()
    cfg = _cfg(args)

    if args.cmd == "preprocess":
        from .data import preprocess_dataset

        preprocess_dataset(cfg, args.cases)

    elif args.cmd == "split":
        from .data import discover_cases, make_split, split_from_nnunet

        if args.from_nnunet:
            split = split_from_nnunet(Path(args.from_nnunet), args.fold)
        else:
            ids = [c["case_id"] for c in discover_cases(C.raw_dir(cfg))]
            split = make_split(ids, args.n_val, args.n_test, args.seed if args.seed is not None else int(cfg["seed"]))
        path = Path(cfg["data"]["split_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split, indent=2))
        # identical split for an nnU-Net baseline (copy over nnUNet_preprocessed/<ds>/splits_final.json)
        nn = path.with_name("splits_final_for_nnunet.json")
        nn.write_text(json.dumps([{"train": split["train"], "val": split["val"]}], indent=2))
        print(f"train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])} -> {path}\n"
              f"nnU-Net baseline with the same split: {nn} (use fold 0)")

    elif args.cmd == "train":
        from .trainer import train_stage

        train_stage(cfg, args.stage, resume=args.resume)

    elif args.cmd == "priors":
        from .pipeline import generate_priors

        generate_priors(cfg, _ckpt(cfg, 1, args.stage1), device=args.device)

    elif args.cmd == "evaluate":
        from .pipeline import evaluate

        evaluate(cfg, _ckpt(cfg, 1, args.stage1), _ckpt(cfg, 2, args.stage2), args.split, args.out,
                 args.save_predictions, args.device)

    elif args.cmd == "predict":
        from .inference import EPCNetPredictor

        pred = EPCNetPredictor(args.stage1, args.stage2, cfg, device=args.device, overlap=args.overlap,
                               roi_margin_mm=args.roi_margin)
        src = Path(args.input)
        if src.is_dir() and any(src.glob("*.nii*")):
            files = sorted(p for p in src.iterdir() if p.name.endswith((".nii", ".nii.gz")))
        else:
            files = [src]  # single file or DICOM series folder
        for f in files:
            res = pred.predict_to_files(f, args.output, save_orbit=not args.no_orbit, stl=args.stl)
            print(f"{f.name}: {res['timings']['total']:.1f}s -> {res['files']}")

    elif args.cmd == "export":
        from .pipeline import export_release

        from . import __version__ as v

        out = Path(args.out or f"release/EPC-Net-weights-v{args.release_version or v}")
        print(export_release(cfg, _ckpt(cfg, 1, args.stage1), _ckpt(cfg, 2, args.stage2), out,
                             args.release_version, not args.no_zip))


if __name__ == "__main__":
    main()
