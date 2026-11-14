from __future__ import annotations

import argparse
import csv
import importlib
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm

from configs.common.config import load_args
from dataloader.paired_image.load_data import create_dataset
from utils.common import CosineAnnealingWarmRestarts, mkdir
from utils.metric import create_metrics


def _write_metrics_row(csv_path: str, row: dict[str, object]) -> None:
    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _truncate_metrics_csv(csv_path: str, *, keep_through_epoch: int) -> None:
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        rows = list(r)
    if not fieldnames or not rows:
        return

    kept: list[dict[str, str]] = []
    for row in rows:
        try:
            ep = int(float(row.get("epoch", "nan")))
        except Exception:
            ep = None
        if ep is None or ep <= keep_through_epoch:
            kept.append(row)

    if len(kept) == len(rows):
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in kept:
            w.writerow(row)


def _load_best_psnr_from_metrics(csv_path: str) -> tuple[float, int]:
    if not os.path.exists(csv_path):
        return float("-inf"), 0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    best_psnr = float("-inf")
    best_epoch = 0
    for row in rows:
        try:
            ep = int(float(row.get("epoch", "nan")))
            psnr = float(row.get("psnr", "nan"))
        except Exception:
            continue
        if psnr == psnr and psnr > best_psnr:
            best_psnr = psnr
            best_epoch = ep
    return best_psnr, best_epoch


def _sync_best_checkpoint(nets_dir: str, *, best_epoch: int, best_path: str) -> None:
    if best_epoch <= 0:
        return
    src = os.path.join(nets_dir, "checkpoint_%06d.tar" % best_epoch)
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    shutil.copyfile(src, best_path)


def _depthwise_conv2d(x: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
    """
    Apply a 2D kernel per-channel (depthwise).
    kernel_2d: [k,k] on CPU; moved/cast to x.device/x.dtype inside.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape={tuple(x.shape)!r}")
    b, c, _, _ = x.shape
    k = int(kernel_2d.shape[-1])
    if kernel_2d.shape != (k, k):
        raise ValueError(f"Expected square kernel [k,k], got {tuple(kernel_2d.shape)!r}")
    weight = kernel_2d.to(device=x.device, dtype=x.dtype).view(1, 1, k, k).repeat(c, 1, 1, 1)
    return F.conv2d(x, weight, bias=None, stride=1, padding=k // 2, groups=c)


def laplacian(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return _depthwise_conv2d(x, k)


def sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    gx = _depthwise_conv2d(x, kx)
    gy = _depthwise_conv2d(x, ky)
    return gx, gy


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape={tuple(x.shape)!r}")
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


@torch.no_grad()
def eval_on_testset(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    metric,
    *,
    epoch: int,
    visual_dir: str,
    visual_n: int = 16,
    visual_size: int | None = None,
) -> tuple[float, float, float]:
    args_test = argparse.Namespace(**vars(args))
    args_test.BATCH_SIZE = 1
    test_loader = create_dataset(args_test, data_path=args_test.TEST_DATASET, mode="test")

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    n = 0

    vis_triplets: list[torch.Tensor] = []
    if visual_size is None:
        visual_size = int(getattr(args, "RESIZE_SIZE", getattr(args, "CROP_SIZE", 256)))

    for batch in test_loader:
        in_img = batch["in_img"].to(device)
        label = batch["label"].to(device)

        out = model(in_img).clamp(0, 1)

        cur_lpips, cur_psnr, cur_ssim = metric.compute(out, label)
        total_lpips += float(cur_lpips)
        total_psnr += float(cur_psnr)
        total_ssim += float(cur_ssim)
        n += 1

        if len(vis_triplets) < visual_n:
            in_vis = F.interpolate(in_img, size=(visual_size, visual_size), mode="bilinear", align_corners=False)
            out_vis = F.interpolate(out, size=(visual_size, visual_size), mode="bilinear", align_corners=False)
            gt_vis = F.interpolate(label, size=(visual_size, visual_size), mode="bilinear", align_corners=False)
            triplet = torch.cat([in_vis, out_vis, gt_vis], dim=3)
            vis_triplets.append(triplet.squeeze(0).detach().cpu())

    avg_psnr = total_psnr / max(1, n)
    avg_ssim = total_ssim / max(1, n)
    avg_lpips = total_lpips / max(1, n)

    if vis_triplets:
        os.makedirs(visual_dir, exist_ok=True)
        vis_path = os.path.join(visual_dir, f"epoch_{epoch:06d}.jpg")
        torchvision.utils.save_image(vis_triplets, vis_path, nrow=4)

    return avg_psnr, avg_ssim, avg_lpips


def train_epoch(
    args: argparse.Namespace,
    train_loader,
    model: torch.nn.Module,
    optimizer,
    device: torch.device,
    *,
    epoch: int,
    iters: int,
) -> tuple[float, float, int]:
    model.train()
    tbar = tqdm(train_loader)
    total_loss = 0.0
    lr = optimizer.state_dict()["param_groups"][0]["lr"]

    w_img = float(getattr(args, "LAMBDA_IMG", 1.0))
    w_base = float(getattr(args, "LAMBDA_BASE", 0.7))
    w_detail = float(getattr(args, "LAMBDA_DETAIL", 0.7))
    w_hp = float(getattr(args, "LAMBDA_HP", 0.3))
    w_grad = float(getattr(args, "LAMBDA_GRAD", 0.2))
    w_mask = float(getattr(args, "LAMBDA_MASK", 0.05))
    w_mask_l1 = float(getattr(args, "MASK_L1", 1.0))
    w_mask_tv = float(getattr(args, "MASK_TV", 1.0))

    need_aux = (w_base > 0.0) or (w_detail > 0.0) or (w_mask > 0.0)

    for batch_idx, batch in enumerate(tbar):
        in_img = batch["in_img"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if need_aux:
            out, aux = model(in_img, return_aux=True)
        else:
            out = model(in_img)
            aux = None

        base_gt = None
        detail_gt = None
        if (w_base > 0.0) or (w_detail > 0.0):
            with torch.no_grad():
                blur = getattr(model, "blur", None)
                if blur is None:
                    raise AttributeError("Model must expose `.blur` when LAMBDA_BASE/LAMBDA_DETAIL are enabled.")
                base_gt = blur(label)
                detail_gt = label - base_gt

        loss_img = F.l1_loss(out, label)
        loss_base = torch.tensor(0.0, device=loss_img.device)
        loss_detail = torch.tensor(0.0, device=loss_img.device)
        if w_base > 0.0:
            if aux is None:
                raise RuntimeError("need_aux=True but aux is None (unexpected).")
            loss_base = F.l1_loss(aux.base_hat, base_gt)  # type: ignore[arg-type]
        if w_detail > 0.0:
            if aux is None:
                raise RuntimeError("need_aux=True but aux is None (unexpected).")
            loss_detail = F.l1_loss(aux.detail_hat, detail_gt)  # type: ignore[arg-type]

        loss_hp = torch.tensor(0.0, device=loss_img.device)
        if w_hp > 0.0:
            loss_hp = F.l1_loss(laplacian(out), laplacian(label))
        loss_grad = torch.tensor(0.0, device=loss_img.device)
        if w_grad > 0.0:
            gx_out, gy_out = sobel_xy(out)
            gx_gt, gy_gt = sobel_xy(label)
            loss_grad = F.l1_loss(gx_out, gx_gt) + F.l1_loss(gy_out, gy_gt)

        loss_m = torch.tensor(0.0, device=loss_img.device)
        if w_mask > 0.0:
            if aux is None:
                raise RuntimeError("need_aux=True but aux is None (unexpected).")
            loss_m = w_mask_l1 * aux.mask.abs().mean() + w_mask_tv * tv_loss(aux.mask)

        loss = (
            w_img * loss_img
            + w_base * loss_base
            + w_detail * loss_detail
            + w_hp * loss_hp
            + w_grad * loss_grad
            + w_mask * loss_m
        )

        loss.backward()
        optimizer.step()

        iters += 1
        total_loss += float(loss.item())
        avg_loss = total_loss / (batch_idx + 1)
        tbar.set_description(f"Training  : Epoch {epoch}, lr {lr:.7f}, Avg. Loss = {avg_loss:.6f}")

    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    return lr, avg_loss, iters


def init(args: argparse.Namespace) -> torch.device:
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "logs")
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "net_checkpoints")
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "train_visual")
    mkdir(args.LOGS_DIR)
    mkdir(args.NETS_DIR)
    mkdir(args.VISUALS_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % int(args.GPU_ID)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(int(args.SEED))
    np.random.seed(int(args.SEED))
    torch.manual_seed(int(args.SEED))
    torch.cuda.manual_seed_all(int(args.SEED))
    if int(args.SEED) == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return device


def load_checkpoint(args: argparse.Namespace, model: torch.nn.Module, optimizer, load_epoch: int) -> tuple[float, int]:
    load_path = os.path.join(args.NETS_DIR, "checkpoint_%06d.tar" % load_epoch)
    print("Loading checkpoint %s" % load_path)
    ckpt = torch.load(load_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    learning_rate = float(ckpt.get("learning_rate", optimizer.state_dict()["param_groups"][0]["lr"]))
    iters = int(ckpt.get("iters", 0))
    print("Learning rate recorded from the checkpoint: %s" % str(learning_rate))
    return learning_rate, iters


def _load_model_from_pkg(args: argparse.Namespace, *, model_pkg: str) -> torch.nn.Module:
    """
    Dynamically load `models.<model_pkg>.net.FaDeNet`.
    """
    mod = importlib.import_module(f"models.{model_pkg}.net")
    cls = getattr(mod, "FaDeNet", None)
    if cls is None:
        raise AttributeError(f"models.{model_pkg}.net must define `FaDeNet`.")
    return cls(
        base_channels=int(getattr(args, "BASE_CHANNELS", 48)),
        num_blocks=tuple(int(x) for x in getattr(args, "NUM_BLOCKS", [2, 2, 2, 2])),
        pad_multiple=int(getattr(args, "PAD_MULTIPLE", 8)),
        blur_kernel_size=int(getattr(args, "BLUR_KERNEL_SIZE", 5)),
        blur_sigma=float(getattr(args, "BLUR_SIGMA", 1.2)),
        detail_alpha=float(getattr(args, "DETAIL_ALPHA", 0.35)),
        fft_gate_scales=tuple(bool(x) for x in getattr(args, "FFT_GATE_SCALES", [False, False, True, True])),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train FaDeNet.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/FaDeNet/prism_config.yaml",
        help="Config YAML (default: configs/FaDeNet/prism_config.yaml).",
    )
    p.add_argument(
        "--model_pkg",
        type=str,
        default="FaDeNet",
        help="Model package under models/.",
    )
    args_cli = p.parse_args()

    args = load_args(["--config", args_cli.config], default_config=args_cli.config)
    device = init(args)

    metric = create_metrics(args, device=device)

    model = _load_model_from_pkg(args, model_pkg=args_cli.model_pkg).to(device)

    optimizer = optim.Adam(
        [{"params": model.parameters(), "initial_lr": float(args.BASE_LR)}],
        lr=float(args.BASE_LR),
        betas=(float(getattr(args, "BETA1", 0.9)), 0.999),
        weight_decay=float(getattr(args, "WEIGHT_DECAY", 0.0)),
    )
    learning_rate = float(args.BASE_LR)
    iters = 0

    load_epoch = int(getattr(args, "LOAD_EPOCH", 0) or 0)
    if load_epoch > 0:
        learning_rate, iters = load_checkpoint(args, model, optimizer, load_epoch)

    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(args.T_0),
        T_mult=int(args.T_MULT),
        eta_min=float(args.ETA_MIN),
        last_epoch=load_epoch - 1,
    )

    train_loader = create_dataset(args, data_path=args.TRAIN_DATASET, mode="train")

    test_visual_dir = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "test_visual")
    mkdir(test_visual_dir)
    metrics_csv = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "metrics.csv")
    if load_epoch > 0:
        _truncate_metrics_csv(metrics_csv, keep_through_epoch=load_epoch)

    best_psnr, best_epoch = _load_best_psnr_from_metrics(metrics_csv)
    best_path = os.path.join(args.NETS_DIR, "best_psnr.checkpoint")
    _sync_best_checkpoint(args.NETS_DIR, best_epoch=best_epoch, best_path=best_path)

    print("****start training!!!****")
    for epoch in range(load_epoch + 1, int(args.EPOCHS) + 1):
        learning_rate, avg_train_loss, iters = train_epoch(args, train_loader, model, optimizer, device, epoch=epoch, iters=iters)
        lr_scheduler.step()

        avg_psnr, avg_ssim, avg_lpips = eval_on_testset(
            args,
            model,
            device,
            metric,
            epoch=epoch,
            visual_dir=test_visual_dir,
            visual_n=16,
        )
        print("Eval     : Epoch %d, PSNR %.4f, SSIM %.4f, LPIPS %.4f" % (epoch, avg_psnr, avg_ssim, avg_lpips))

        ckpt = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "iters": iters,
            "optimizer": optimizer.state_dict(),
            "state_dict": model.state_dict(),
            "train_loss": avg_train_loss,
            "test_psnr": avg_psnr,
            "test_ssim": avg_ssim,
            "test_lpips": avg_lpips,
        }
        epoch_ckpt_path = os.path.join(args.NETS_DIR, "checkpoint_%06d.tar" % epoch)
        torch.save(ckpt, epoch_ckpt_path)

        is_best = False
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            is_best = True
            torch.save(ckpt, best_path)

        latest_ckpt_path = os.path.join(args.NETS_DIR, "checkpoint_latest.tar")
        torch.save(ckpt, latest_ckpt_path)

        _write_metrics_row(
            metrics_csv,
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": avg_train_loss,
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "lpips": avg_lpips,
                "checkpoint": os.path.basename(epoch_ckpt_path),
                "is_best_psnr": int(is_best),
                "best_psnr": best_psnr,
                "best_epoch": best_epoch,
            },
        )


if __name__ == "__main__":
    main()
