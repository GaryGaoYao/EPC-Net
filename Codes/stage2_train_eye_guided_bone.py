#!/usr/bin/env python3
"""Stage 2: train eye-guided orbital-bone segmentation with SegResNet.

Expected dataset layout::

    DATASET_ROOT/
    ├── imagesTr/case001_0000.nii.gz
    ├── labelsTr/case001.nii.gz
    └── eyePredTr/case001_0000_pred_origspace.nii.gz

The network receives two channels: normalized CT and an eye-distance guidance map.
The segmentation loss combines eye-weighted soft Dice and cross-entropy.
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
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai import __version__ as monai_version
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)
from monai.utils import set_determinism
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    eye_dir: str | None
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
    warmup_epochs: int
    warmup_start_factor: float
    plateau_patience: int
    plateau_factor: float
    min_learning_rate: float
    grad_clip_norm: float
    sigma_mm: float
    alpha: float
    band_min_mm: float
    band_max_mm: float
    dilate_mm: float
    dice_weight: float
    ce_weight: float
    amp: bool
    resume: str | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--eye-dir",
        default=None,
        help="Eye-prediction directory. Defaults to DATA_DIR/eyePredTr.",
    )
    parser.add_argument("--output-dir", default="outputs/stage2")
    parser.add_argument("--run-name", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(128, 128, 128), metavar=("X", "Y", "Z"))
    parser.add_argument("--spacing", type=float, nargs=3, default=(0.5, 0.5, 1.0), metavar=("SX", "SY", "SZ"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--sw-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=12.0)
    parser.add_argument("--sigma-mm", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--band-min-mm", type=float, default=0.0)
    parser.add_argument("--band-max-mm", type=float, default=25.0)
    parser.add_argument("--dilate-mm", type=float, default=1.5)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", default=None, help="Path to a full checkpoint created by this script.")
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1.")
    if any(value <= 0 for value in args.spacing):
        parser.error("All spacing values must be positive.")
    if args.sigma_mm <= 0 or args.band_max_mm < args.band_min_mm:
        parser.error("Invalid eye-attention distance parameters.")
    if not 0.0 < args.warmup_start_factor <= 1.0:
        parser.error("--warmup-start-factor must be in (0, 1].")
    if not 0.0 < args.plateau_factor < 1.0:
        parser.error("--plateau-factor must be in (0, 1).")

    return TrainConfig(
        data_dir=args.data_dir,
        eye_dir=args.eye_dir,
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
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        min_learning_rate=args.min_learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        sigma_mm=args.sigma_mm,
        alpha=args.alpha,
        band_min_mm=args.band_min_mm,
        band_max_mm=args.band_max_mm,
        dilate_mm=args.dilate_mm,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        amp=not args.no_amp,
        resume=args.resume,
    )


def image_case_id(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected image name: {path.name}")
    return path.name[: -len(suffix)]


def label_case_id(path: Path) -> str:
    suffix = ".nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected label name: {path.name}")
    return path.name[: -len(suffix)]


def collect_cases(data_dir: Path, eye_dir: Path) -> list[dict[str, str]]:
    images = sorted((data_dir / "imagesTr").glob("*_0000.nii.gz"))
    labels = sorted((data_dir / "labelsTr").glob("*.nii.gz"))
    if not images:
        raise FileNotFoundError(f"No CT images found in {data_dir / 'imagesTr'}")
    if not labels:
        raise FileNotFoundError(f"No labels found in {data_dir / 'labelsTr'}")
    if not eye_dir.is_dir():
        raise FileNotFoundError(f"Eye-prediction directory not found: {eye_dir}")

    label_map = {label_case_id(path): path for path in labels}
    cases: list[dict[str, str]] = []
    problems: list[str] = []
    for image in images:
        case_id = image_case_id(image)
        label = label_map.get(case_id)
        eye = eye_dir / f"{case_id}_0000_pred_origspace.nii.gz"
        if label is None:
            problems.append(f"{case_id}: missing label")
            continue
        if not eye.is_file():
            problems.append(f"{case_id}: missing eye prediction ({eye.name})")
            continue
        cases.append(
            {"case_id": case_id, "image": str(image), "label": str(label), "eye": str(eye)}
        )

    unused_labels = sorted(set(label_map) - {image_case_id(path) for path in images})
    problems.extend(f"{case_id}: label has no matching image" for case_id in unused_labels)
    if problems:
        preview = "\n".join(problems[:20])
        raise RuntimeError(f"Dataset matching failed ({len(problems)} issue(s)):\n{preview}")
    if len(cases) < 2:
        raise RuntimeError("At least two matched cases are required.")
    return cases


def split_cases(
    cases: list[dict[str, str]], val_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


class MakeEyeAttentiond(MapTransform):
    """Create eye guidance, spatial weights, and a sampling band in physical units."""

    def __init__(
        self,
        eye_key: Hashable,
        spacing_xyz: Sequence[float],
        guidance_key: Hashable = "guidance",
        weight_key: Hashable = "weight",
        band_key: Hashable = "eye_band",
        sigma_mm: float = 5.0,
        alpha: float = 6.0,
        band_min_mm: float = 0.0,
        band_max_mm: float = 25.0,
        dilate_mm: float = 0.0,
    ) -> None:
        super().__init__(keys=(eye_key,))
        self.eye_key = eye_key
        self.guidance_key = guidance_key
        self.weight_key = weight_key
        self.band_key = band_key
        # MONAI pixdim values map directly to the tensor spatial dimensions.
        self.spatial_spacing = tuple(float(value) for value in spacing_xyz)
        self.sigma_mm = float(sigma_mm)
        self.alpha = float(alpha)
        self.band_min_mm = float(band_min_mm)
        self.band_max_mm = float(band_max_mm)
        self.dilate_mm = float(dilate_mm)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        result = dict(data)
        eye = torch.as_tensor(result[self.eye_key])
        eye_array = eye.detach().cpu().numpy()
        if eye_array.ndim == 4:
            eye_array = eye_array[0]
        if eye_array.ndim != 3:
            raise ValueError(f"Expected eye mask [1,D,H,W] or [D,H,W], got {eye.shape}")

        eye_mask = eye_array > 0.5
        shape = eye_mask.shape
        if not np.any(eye_mask):
            result[self.guidance_key] = torch.zeros((1, *shape), dtype=torch.float32)
            result[self.weight_key] = torch.ones((1, *shape), dtype=torch.float32)
            result[self.band_key] = torch.zeros((1, *shape), dtype=torch.uint8)
            return result

        if self.dilate_mm > 0:
            distance_to_eye = distance_transform_edt(~eye_mask, sampling=self.spatial_spacing)
            eye_mask = distance_to_eye <= self.dilate_mm

        distance_mm = distance_transform_edt(~eye_mask, sampling=self.spatial_spacing).astype(np.float32)
        guidance = np.exp(-distance_mm / self.sigma_mm).astype(np.float32)
        weight = (
            1.0
            + self.alpha
            * np.exp(-(distance_mm**2) / (2.0 * self.sigma_mm**2))
        ).astype(np.float32)
        band = (
            (distance_mm >= self.band_min_mm) & (distance_mm <= self.band_max_mm)
        ).astype(np.uint8)

        result[self.guidance_key] = torch.from_numpy(guidance[None])
        result[self.weight_key] = torch.from_numpy(weight[None])
        result[self.band_key] = torch.from_numpy(band[None])
        return result


def weighted_soft_dice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    classes = probabilities.shape[1]
    target_one_hot = F.one_hot(
        target.squeeze(1).long(), num_classes=classes
    ).permute(0, 4, 1, 2, 3).float()
    class_weight = weight.expand(-1, classes, -1, -1, -1)
    intersection = (probabilities * target_one_hot * class_weight).sum(dim=(2, 3, 4))
    denominator = (
        (probabilities * class_weight).sum(dim=(2, 3, 4))
        + (target_one_hot * class_weight).sum(dim=(2, 3, 4))
    )
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice[:, 1:].mean()


class EyeWeightedDiceCELoss(torch.nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        dice = weighted_soft_dice_loss(probabilities, target, weight)
        cross_entropy = F.cross_entropy(
            logits, target.squeeze(1).long(), reduction="none"
        )
        voxel_weight = weight.squeeze(1)
        weighted_ce = (cross_entropy * voxel_weight).sum() / voxel_weight.sum().clamp_min(1e-6)
        return self.dice_weight * dice + self.ce_weight * weighted_ce


def build_transforms(config: TrainConfig) -> tuple[Compose, Compose]:
    common = [
        LoadImaged(keys=("image", "label", "eye")),
        EnsureChannelFirstd(keys=("image", "label", "eye")),
        Orientationd(keys=("image", "label", "eye"), axcodes="RAS"),
        Spacingd(
            keys=("image", "label", "eye"),
            pixdim=config.spacing,
            mode=("bilinear", "nearest", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        EnsureTyped(keys=("image", "label", "eye"), track_meta=False),
        MakeEyeAttentiond(
            eye_key="eye",
            spacing_xyz=config.spacing,
            sigma_mm=config.sigma_mm,
            alpha=config.alpha,
            band_min_mm=config.band_min_mm,
            band_max_mm=config.band_max_mm,
            dilate_mm=config.dilate_mm,
        ),
        ConcatItemsd(keys=("image", "guidance"), name="model_input", dim=0),
    ]

    train_transform = Compose(
        common
        + [
            SpatialPadd(
                keys=("model_input", "label", "weight", "eye_band"),
                spatial_size=config.roi_size,
            ),
            RandCropByPosNegLabeld(
                keys=("model_input", "label", "weight", "eye_band"),
                label_key="eye_band",
                spatial_size=config.roi_size,
                pos=3,
                neg=1,
                num_samples=1,
                image_key="model_input",
                image_threshold=0.0,
            ),
            RandFlipd(
                keys=("model_input", "label", "weight", "eye_band"),
                prob=0.5,
                spatial_axis=0,
            ),
            RandRotate90d(
                keys=("model_input", "label", "weight", "eye_band"),
                prob=0.5,
                max_k=3,
            ),
            EnsureTyped(keys=("model_input", "label", "weight"), track_meta=False),
            SelectItemsd(keys=("model_input", "label", "weight")),
        ]
    )
    val_transform = Compose(
        common
        + [
            EnsureTyped(keys=("model_input", "label", "weight"), track_meta=False),
            SelectItemsd(keys=("model_input", "label", "weight")),
        ]
    )
    return train_transform, val_transform


def build_loader(
    files: list[dict[str, str]], transform: Compose, config: TrainConfig, training: bool
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
        in_channels=2,
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
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
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
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "best_epoch": best_epoch,
            "config": asdict(config),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


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
            images = batch["model_input"].to(device, non_blocking=True)
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

    data_dir = Path(config.data_dir).expanduser()
    eye_dir = Path(config.eye_dir).expanduser() if config.eye_dir else data_dir / "eyePredTr"
    cases = collect_cases(data_dir, eye_dir)
    train_files, val_files = split_cases(cases, config.val_fraction, config.seed)
    train_transform, val_transform = build_transforms(config)
    train_loader = build_loader(train_files, train_transform, config, training=True)
    val_loader = build_loader(val_files, val_transform, config, training=False)

    model = build_model().to(device)
    loss_function = EyeWeightedDiceCELoss(config.dice_weight, config.ce_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.plateau_factor,
        patience=config.plateau_patience,
        threshold=1e-4,
        min_lr=config.min_learning_rate,
    )
    scaler = make_grad_scaler(amp_enabled)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    latest_checkpoint = run_dir / "latest_checkpoint.pth"
    best_checkpoint = run_dir / "best_checkpoint.pth"
    latest_weights = run_dir / "latest_model_weights.pth"
    best_weights = run_dir / "best_model_weights.pth"
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
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
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
        "resolved_eye_dir": str(eye_dir),
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

        if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
            initial_lr = config.learning_rate * config.warmup_start_factor
            warmup_progress = float(epoch + 1) / float(config.warmup_epochs)
            set_learning_rate(
                optimizer,
                initial_lr + (config.learning_rate - initial_lr) * warmup_progress,
            )

        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            images = batch["model_input"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = loss_function(logits, labels, weights)

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
            if epoch + 1 > config.warmup_epochs:
                scheduler.step(val_dice)
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                atomic_torch_save(model.state_dict(), best_weights)
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    scheduler,
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
            scheduler,
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
            f"Epoch {epoch + 1:04d}/{config.max_epochs} | lr={learning_rate:.3e} | "
            f"loss={train_loss:.4f} | val_dice={val_text} | "
            f"best={best_dice:.4f}@{best_epoch} | {elapsed:.1f}s"
        )

    print(f"Training complete. Best validation Dice: {best_dice:.4f} at epoch {best_epoch}.")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
