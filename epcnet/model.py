"""MONAI 3D SegResNet backbone (Sec. 3.6) and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import SegResNet


def build_segresnet(in_channels: int, out_channels: int = 2, model_cfg: dict | None = None) -> SegResNet:
    m = model_cfg or {}
    return SegResNet(
        spatial_dims=3,
        init_filters=int(m.get("init_filters", 32)),
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        dropout_prob=float(m.get("dropout_prob", 0.2)),
        norm=("GROUP", {"num_groups": int(m.get("num_groups", 8))}),
        blocks_down=tuple(m.get("blocks_down", (1, 2, 2, 4))),
        blocks_up=tuple(m.get("blocks_up", (1, 1, 1))),
        upsample_mode="nontrainable",  # 1x1x1 conv + interpolation upsampling
    )


def load_torch(path: str | Path, map_location="cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(str(path), map_location=map_location)


def extract_state_dict(obj: Any) -> dict:
    """Accept a raw state_dict or any checkpoint dict produced by this or the older scripts."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "network_weights"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not obj:
        raise ValueError("No network weights found in checkpoint")
    return {k[7:] if k.startswith("module.") else k: v for k, v in obj.items()}


def in_channels_from_state_dict(state_dict: dict) -> int:
    return int(state_dict["convInit.conv.weight"].shape[1])


def load_model(path: str | Path, model_cfg: dict | None = None, device: str | torch.device = "cpu") -> tuple[SegResNet, dict]:
    obj = load_torch(path, map_location="cpu")
    sd = extract_state_dict(obj)
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    # the architecture stored with the weights wins over any config; width is read from the tensors
    cfg = dict(meta.get("model_cfg") or model_cfg or {})
    cfg["init_filters"] = int(sd["convInit.conv.weight"].shape[0])
    out_ch = int(sd["conv_final.2.conv.weight"].shape[0]) if "conv_final.2.conv.weight" in sd else 2
    net = build_segresnet(in_channels_from_state_dict(sd), out_ch, cfg)
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval(), meta
