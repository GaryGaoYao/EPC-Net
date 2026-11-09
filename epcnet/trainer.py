"""Training loop shared by Stage I and Stage II (Sec. 3.5, 3.10, Table 2).

* AdamW (lr 1e-4, wd 1e-5), cosine annealing, batch size 2, up to 500 epochs
* validation every epoch: full-volume sliding-window inference, mean foreground DSC
* best checkpoint = highest validation DSC, ties broken by lower HD95
* early stopping after 50 epochs without improvement
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader

from . import __version__
from . import config as C
from .augment import GpuAugment
from .data import PatchDataset, load_case_volume, load_split, worker_init_fn
from .losses import WeightedDiceCELoss
from .metrics import hd95
from .model import build_segresnet, load_torch
from .priors import guidance_map, weight_map

log = logging.getLogger("epcnet")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: Path | None = None) -> None:
    root = logging.getLogger("epcnet")
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(net, cfg: dict, stage: int, case_ids, device, amp: bool, cache: dict | None = None) -> list[dict]:
    sc = C.stage_config(cfg, stage)
    vdev = torch.device(sc.get("val_device", "cuda")) if device.type == "cuda" else device
    net.eval()
    results = []
    for cid in case_ids:
        vol = cache[cid] if cache is not None and cid in cache else load_case_volume(cfg, stage, cid)
        if cache is not None and cid not in cache:
            vol["image"] = vol["image"].half()  # keep RAM usage low; cast back per window
            cache[cid] = vol
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = sliding_window_inference(
                vol["image"], roi_size=sc["patch_size"], sw_batch_size=int(sc["sw_batch_size"]), predictor=lambda t: net(t.float()),
                overlap=float(sc["sw_overlap"]), mode="gaussian", sw_device=device, device=vdev,
            )
        pred = logits.argmax(1)[0]
        gt = vol["label"].to(pred.device).bool()
        p = pred.bool()
        denom = p.sum() + gt.sum()
        dsc = 1.0 if denom == 0 else float(2.0 * (p & gt).sum() / denom)
        results.append({"case_id": cid, "dsc": dsc, "pred": p.cpu().numpy(), "gt": gt.cpu().numpy(),
                        "spacing": vol["spacing"]})
        del logits
    return results


def _mean_hd95(res: list[dict]) -> float:
    return float(np.nanmean([hd95(r["pred"], r["gt"], r["spacing"]) for r in res]))


def train_stage(cfg: dict, stage: int, resume: bool = False) -> Path:
    sc = C.stage_config(cfg, stage)
    out = C.run_dir(cfg, stage)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "train.log")
    C.save_config(cfg, out / "config.yaml")
    seed_everything(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(sc["amp"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True          # fixed patch size -> fastest conv algorithms
        torch.backends.cuda.matmul.allow_tf32 = True   # Ampere (RTX A6000) tensor cores
        torch.backends.cudnn.allow_tf32 = True

    split = load_split(cfg)
    train_ids, val_ids = split["train"], split["val"]
    if stage == 2:
        in_ch = 2 if sc.get("use_guidance", True) else 1
        weighted = bool(sc.get("use_weighted_loss", True))
    else:
        in_ch, weighted = int(sc["in_channels"]), False
    log.info("EPC-Net %s | stage %d | device %s | AMP %s | train %d / val %d cases | in_channels %d | weighted loss %s",
             __version__, stage, device, amp, len(train_ids), len(val_ids), in_ch, weighted)

    ds = PatchDataset(cfg, stage, train_ids)
    nw = int(cfg["data"]["num_workers"])
    loader = DataLoader(ds, batch_size=int(sc["batch_size"]), shuffle=True, num_workers=nw, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                        prefetch_factor=4 if nw > 0 else None, worker_init_fn=worker_init_fn if nw > 0 else None)
    augment = GpuAugment(cfg["augment"], sc["patch_size"], ds.big, sc["spacing"])
    use_guidance = stage == 2 and bool(sc.get("use_guidance", True))
    prior = cfg["prior"]

    net = build_segresnet(in_ch, int(sc["out_channels"]), cfg["model"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(sc["lr"]), weight_decay=float(sc["weight_decay"]))
    max_epochs = int(sc["max_epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=float(sc["eta_min"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = WeightedDiceCELoss()

    start, best_dsc, best_hd, best_epoch = 0, -1.0, float("nan"), 0
    last_ckpt, best_ckpt = out / "last_checkpoint.pt", out / "best_model.pt"
    if resume and last_ckpt.exists():
        st = load_torch(last_ckpt)
        net.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start, best_dsc, best_hd, best_epoch = st["epoch"], st["best_dsc"], st["best_hd95"], st["best_epoch"]
        log.info("Resumed at epoch %d (best DSC %.4f @ %d)", start, best_dsc, best_epoch)

    log_csv = out / "log.csv"
    if not (resume and log_csv.exists()):
        with log_csv.open("w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "lr", "train_loss", "val_dsc", "val_hd95", "best_dsc", "best_epoch", "seconds"])

    meta = {
        "epcnet_version": __version__, "stage": stage, "in_channels": in_ch, "out_channels": int(sc["out_channels"]),
        "spacing": sc["spacing"], "patch_size": sc["patch_size"], "model_cfg": cfg["model"],
        "preprocess": cfg["preprocess"], "prior": cfg["prior"] if stage == 2 else None,
        "use_guidance": bool(sc.get("use_guidance", False)), "use_weighted_loss": weighted,
        "train_cases": train_ids, "val_cases": val_ids, "torch": str(torch.__version__), "python": platform.python_version(),
    }
    val_cache: dict = {}
    patience, val_interval = int(sc["patience"]), int(sc["val_interval"])
    tol = float(sc["tie_tolerance"])

    for epoch in range(start, max_epochs):
        t0 = time.time()
        net.train()
        losses = []
        for batch in loader:
            dist = batch["dist"].to(device, non_blocking=True) if "dist" in batch else None
            x, y, dist = augment(batch["image"].to(device, non_blocking=True),
                                 batch["label"].to(device, non_blocking=True), dist)
            if use_guidance:   # feature prior G as second input channel
                x = torch.cat([x, guidance_map(dist, prior["sigma"], prior["delta"])], 1)
            w = weight_map(dist, prior["sigma"], prior["alpha"], prior["delta"]) if weighted else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = net(x)
            loss = loss_fn(logits, y, w)
            scaler.scale(loss).backward()
            if sc.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(sc["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        sched.step()
        train_loss = float(np.mean(losses)) if losses else math.nan

        val_dsc = val_hd = math.nan
        if (epoch + 1) % val_interval == 0 or epoch + 1 == max_epochs:
            res = validate(net, cfg, stage, val_ids, device, amp, val_cache)
            val_dsc = float(np.mean([r["dsc"] for r in res]))
            improved = val_dsc > best_dsc + tol
            if not improved and abs(val_dsc - best_dsc) <= tol:  # tie -> lower HD95 wins
                val_hd = _mean_hd95(res)
                if math.isnan(best_hd) or math.isinf(best_hd):  # HD95 of the stored best model (computed lazily)
                    best_net = build_segresnet(in_ch, int(sc["out_channels"]), cfg["model"]).to(device)
                    best_net.load_state_dict(load_torch(best_ckpt)["state_dict"])
                    best_hd = _mean_hd95(validate(best_net, cfg, stage, val_ids, device, amp, val_cache))
                    del best_net
                improved = val_hd < best_hd
            if improved:
                best_dsc, best_hd, best_epoch = val_dsc, val_hd, epoch + 1
                _atomic_save({"state_dict": net.state_dict(), "meta": {**meta, "epoch": best_epoch,
                              "val_dsc": best_dsc, "val_hd95": best_hd,
                              "val_per_case": {r["case_id"]: r["dsc"] for r in res}}}, best_ckpt)
            del res

        _atomic_save({"state_dict": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_dsc": best_dsc, "best_hd95": best_hd,
                      "best_epoch": best_epoch}, last_ckpt)
        dt = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        with log_csv.open("a", newline="") as fh:
            csv.writer(fh).writerow([epoch + 1, f"{lr:.3e}", f"{train_loss:.5f}", f"{val_dsc:.5f}", f"{val_hd:.3f}",
                                     f"{best_dsc:.5f}", best_epoch, f"{dt:.1f}"])
        log.info("stage%d ep %03d/%d | loss %.4f | val DSC %.4f | best %.4f @%d | lr %.2e | %.0fs",
                 stage, epoch + 1, max_epochs, train_loss, val_dsc, best_dsc, best_epoch, lr, dt)
        if epoch + 1 - best_epoch >= patience:
            log.info("Early stopping: no validation improvement for %d epochs.", patience)
            break

    if best_epoch > 0 and (math.isnan(best_hd) or math.isinf(best_hd)):  # report HD95 of the selected model
        ck = load_torch(best_ckpt)
        net.load_state_dict(ck["state_dict"])
        best_hd = _mean_hd95(validate(net, cfg, stage, val_ids, device, amp, val_cache))
        ck["meta"]["val_hd95"] = best_hd
        _atomic_save(ck, best_ckpt)
    summary = {"best_epoch": best_epoch, "best_val_dsc": best_dsc, "best_val_hd95": best_hd, "checkpoint": str(best_ckpt)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done. Best val DSC %.4f (HD95 %.2f mm) at epoch %d -> %s", best_dsc, best_hd, best_epoch, best_ckpt)
    return best_ckpt
