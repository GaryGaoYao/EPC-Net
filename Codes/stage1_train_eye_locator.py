#!/usr/bin/env python3
"""Stage 1: train a 3D SegResNet eye-region locator.

Expected dataset layout:

    DATASET_ROOT/
    ├── imagesTr/
    │   ├── case001_0000.nii.gz
    │   └── ...
    └── labelsTr/
        ├── case001.nii.gz
        └── ...

The script keeps the original model and preprocessing strategy while adding:
case-ID-based matching, deterministic splitting, complete checkpoints, safe AMP,
configuration/split manifests, and reproducible CSV logging.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai import __version__ as monai_version
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
)
from monai.utils import set_determinism


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    output_dir: str
    run_name: str
    gpu: str
    seed: int
    val_fraction: float
    max_epochs: int
    val_interval: int
    batch_size: int
    roi_size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    num_workers: int
    cache_rate: float
    sw_batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    amp: bool
    resume: str | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Dataset root containing imagesTr and labelsTr.")
    parser.add_argument("--output-dir", default="outputs/stage1", help="Root directory for training runs.")
    parser.add_argument("--run-name", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--gpu", default="0", help="CUDA device index, or an empty string for default visibility.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96), metavar=("X", "Y", "Z"))
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("SX", "SY", "SZ"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--sw-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="0 disables gradient clipping.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--resume", default=None, help="Path to a full checkpoint created by this script.")
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1.")
    if args.max_epochs < 1 or args.val_interval < 1:
        parser.error("--max-epochs and --val-interval must be positive.")
    if any(v < 1 for v in args.roi_size):
        parser.error("All ROI dimensions must be positive.")
    if any(v <= 0 for v in args.spacing):
        parser.error("All spacing values must be positive.")

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        gpu=args.gpu,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_epochs=args.max_epochs,
        val_interval=args.val_interval,
        batch_size=args.batch_size,
        roi_size=tuple(args.roi_size),
        spacing=tuple(args.spacing),
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        sw_batch_size=args.sw_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        amp=not args.no_amp,
        resume=args.resume,
    )


def case_id_from_image(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected image name: {path.name}")
    return path.name[: -len(suffix)]


def case_id_from_label(path: Path) -> str:
    suffix = ".nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected label name: {path.name}")
    return path.name[: -len(suffix)]


def collect_cases(data_dir: Path) -> list[dict[str, str]]:
    images = sorted((data_dir / "imagesTr").glob("*_0000.nii.gz"))
    labels = sorted((data_dir / "labelsTr").glob("*.nii.gz"))
    if not images:
        raise FileNotFoundError(f"No training images found in {data_dir / 'imagesTr'}")
    if not labels:
        raise FileNotFoundError(f"No training labels found in {data_dir / 'labelsTr'}")

    image_map = {case_id_from_image(path): path for path in images}
    label_map = {case_id_from_label(path): path for path in labels}
    shared = sorted(image_map.keys() & label_map.keys())
    missing_labels = sorted(image_map.keys() - label_map.keys())
    missing_images = sorted(label_map.keys() - image_map.keys())

    if missing_labels or missing_images:
        message = ["Image/label case IDs do not match."]
        if missing_labels:
            message.append(f"Missing labels ({len(missing_labels)}): {missing_labels[:10]}")
        if missing_images:
            message.append(f"Missing images ({len(missing_images)}): {missing_images[:10]}")
        raise RuntimeError("\n".join(message))
    if len(shared) < 2:
        raise RuntimeError("At least two matched cases are required for train/validation splitting.")

    return [
        {"case_id": case_id, "image": str(image_map[case_id]), "label": str(label_map[case_id])}
        for case_id in shared
    ]


def split_cases(
    cases: list[dict[str, str]], val_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def build_transforms(config: TrainConfig) -> tuple[Compose, Compose]:
    common = [
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(keys=("image", "label"), axcodes="RAS"),
        Spacingd(
            keys=("image", "label"),
            pixdim=config.spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        EnsureTyped(keys=("image", "label"), track_meta=False),
    ]

    train_transform = Compose(
        common
        + [
            SpatialPadd(keys=("image", "label"), spatial_size=config.roi_size),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=config.roi_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0.0,
            ),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
            RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3),
        ]
    )
    return train_transform, Compose(common)


def build_loader(
    files: list[dict[str, str]],
    transform: Compose,
    config: TrainConfig,
    training: bool,
) -> DataLoader:
    dataset = CacheDataset(
        data=files,
        transform=transform,
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size if training else 1,
        shuffle=training,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def build_model() -> SegResNet:
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=32,
        dropout_prob=0.2,
    )


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_dice: float,
    best_epoch: int,
    config: TrainConfig,
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "best_epoch": best_epoch,
            "config": asdict(config),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    metric: DiceMetric,
    amp_enabled: bool,
) -> float:
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=config.roi_size,
                    sw_batch_size=config.sw_batch_size,
                    predictor=model,
                    overlap=0.25,
                )
            predictions = torch.argmax(logits, dim=1, keepdim=True)
            metric(
                y_pred=[item for item in decollate_batch(predictions)],
                y=[item for item in decollate_batch(labels)],
            )
    value = float(metric.aggregate().item())
    metric.reset()
    return value


def train(config: TrainConfig) -> None:
    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    set_determinism(seed=config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    resume_path = Path(config.resume).expanduser().resolve() if config.resume else None
    run_dir = resume_path.parent if resume_path else Path(config.output_dir).expanduser() / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(Path(config.data_dir).expanduser())
    train_files, val_files = split_cases(cases, config.val_fraction, config.seed)
    train_transform, val_transform = build_transforms(config)
    train_loader = build_loader(train_files, train_transform, config, training=True)
    val_loader = build_loader(val_files, val_transform, config, training=False)

    model = build_model().to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = make_grad_scaler(amp_enabled)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    latest_checkpoint = run_dir / "latest_checkpoint.pth"
    best_checkpoint = run_dir / "best_checkpoint.pth"
    best_weights = run_dir / "best_model_weights.pth"
    latest_weights = run_dir / "latest_model_weights.pth"
    log_path = run_dir / "training_log.csv"

    start_epoch = 0
    best_dice = -1.0
    best_epoch = -1
    if resume_path:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_dice = float(checkpoint.get("best_dice", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"Resumed from epoch {start_epoch}: {resume_path}")

    if not log_path.exists() or start_epoch == 0:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["epoch", "learning_rate", "train_loss", "val_dice", "best_dice", "seconds"]
            )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "monai_version": monai_version,
        "device": str(device),
        "configuration": asdict(config),
        "training_cases": [item["case_id"] for item in train_files],
        "validation_cases": [item["case_id"] for item in val_files],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Device: {device} | AMP: {amp_enabled}")
    print(f"Training cases: {len(train_files)} | Validation cases: {len(val_files)}")
    print(f"Run directory: {run_dir.resolve()}")

    for epoch in range(start_epoch, config.max_epochs):
        started = time.time()
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            if config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            steps += 1

        train_loss = total_loss / max(steps, 1)
        val_dice: float | None = None
        if (epoch + 1) % config.val_interval == 0 or epoch + 1 == config.max_epochs:
            val_dice = validate(
                model, val_loader, device, config, dice_metric, amp_enabled
            )
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                atomic_torch_save(model.state_dict(), best_weights)
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    best_dice,
                    best_epoch,
                    config,
                )
                print(f"Saved new best model: Dice={best_dice:.4f} at epoch {best_epoch}")

        atomic_torch_save(model.state_dict(), latest_weights)
        save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scaler,
            epoch + 1,
            best_dice,
            best_epoch,
            config,
        )

        elapsed = time.time() - started
        learning_rate = float(optimizer.param_groups[0]["lr"])
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch + 1,
                    f"{learning_rate:.8e}",
                    f"{train_loss:.6f}",
                    "" if val_dice is None else f"{val_dice:.6f}",
                    f"{best_dice:.6f}",
                    f"{elapsed:.2f}",
                ]
            )

        val_text = "not evaluated" if val_dice is None else f"{val_dice:.4f}"
        print(
            f"Epoch {epoch + 1:04d}/{config.max_epochs} | "
            f"loss={train_loss:.4f} | val_dice={val_text} | "
            f"best={best_dice:.4f}@{best_epoch} | {elapsed:.1f}s"
        )

    print(f"Training complete. Best validation Dice: {best_dice:.4f} at epoch {best_epoch}.")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
