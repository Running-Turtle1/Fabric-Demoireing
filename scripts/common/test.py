from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from configs.common.config import load_args
from dataloader.paired_image.load_data import create_dataset
from utils.common import PSNR, SSIM, mkdir


def set_logging(log_path: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def init(args: argparse.Namespace, *, device_arg: str = "") -> torch.device:
    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "test_result")
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, "net_checkpoints")
    mkdir(args.TEST_RESULT_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % int(args.GPU_ID)
    if device_arg:
        device = torch.device(device_arg)
    else:
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


def load_state_dict_any(weights_path: str | Path) -> dict[str, torch.Tensor]:
    obj = torch.load(str(weights_path), map_location="cpu")
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)!r}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}
    return state


def load_model_from_pkg(args: argparse.Namespace, *, model_pkg: str) -> torch.nn.Module:
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


def resolve_checkpoint(args: argparse.Namespace, *, load_path: str, test_epoch: str) -> tuple[str, str]:
    if load_path:
        return load_path, "custom"

    epoch = str(test_epoch or getattr(args, "TEST_EPOCH", "auto"))
    if epoch in {"auto", "latest"}:
        return os.path.join(args.NETS_DIR, "checkpoint_latest.tar"), "latest"
    if epoch == "best":
        return os.path.join(args.NETS_DIR, "best_psnr.checkpoint"), "best"

    epoch_i = int(epoch)
    return os.path.join(args.NETS_DIR, "checkpoint_%06d.tar" % epoch_i), "%04d" % epoch_i


class Metrics:
    def __init__(self, device: torch.device, *, use_lpips: bool) -> None:
        self.psnr = PSNR().to(device).eval()
        self.ssim = SSIM().to(device).eval()
        self.lpips_fn = None
        if use_lpips:
            import lpips

            self.lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[float | None, float, float]:
        pred = pred.clamp(0, 1)
        gt = gt.clamp(0, 1)
        psnr_v = float(self.psnr(pred, gt).item())
        ssim_v = float(self.ssim(pred, gt).item())
        lpips_v = None
        if self.lpips_fn is not None:
            lpips_v = float(self.lpips_fn.forward(pred, gt, normalize=True).item())
        return lpips_v, psnr_v, ssim_v


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def test_loop(
    args: argparse.Namespace,
    test_loader,
    model: torch.nn.Module,
    metrics: Metrics,
    *,
    save_path: Path,
    save_img: str,
    save_triplet: bool,
    max_samples: int,
) -> dict[str, object]:
    device = next(model.parameters()).device
    model.eval()

    rows: list[dict[str, object]] = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    n_lpips = 0
    n = 0

    tbar = tqdm(test_loader)
    for batch in tbar:
        number = batch["number"]
        sample_id = str(number[0] if isinstance(number, (list, tuple)) else number)

        in_img = batch["in_img"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        out = model(in_img).clamp(0, 1)
        lpips_v, psnr_v, ssim_v = metrics.compute(out, label)

        row: dict[str, object] = {
            "sample": sample_id,
            "psnr": psnr_v,
            "ssim": ssim_v,
        }
        if lpips_v is not None:
            row["lpips"] = lpips_v
            total_lpips += lpips_v
            n_lpips += 1
        rows.append(row)

        if save_img:
            out_cpu = out.detach().cpu().clamp(0, 1)
            torchvision.utils.save_image(out_cpu, save_path / f"{sample_id}.{save_img}")

        if save_triplet:
            triplet = torch.cat([in_img, out, label], dim=3).detach().cpu().clamp(0, 1)
            torchvision.utils.save_image(triplet, save_path / f"{sample_id}_input_out_gt.{save_img or 'png'}")

        total_psnr += psnr_v
        total_ssim += ssim_v
        n += 1

        avg_psnr = total_psnr / max(1, n)
        avg_ssim = total_ssim / max(1, n)
        if n_lpips:
            avg_lpips = total_lpips / n_lpips
            tbar.set_description(
                "Test: Avg. LPIPS = %.4f, Avg. PSNR = %.4f, Avg. SSIM = %.4f"
                % (avg_lpips, avg_psnr, avg_ssim)
            )
        else:
            tbar.set_description("Test: Avg. PSNR = %.4f, Avg. SSIM = %.4f" % (avg_psnr, avg_ssim))

        if max_samples > 0 and n >= max_samples:
            break

    _write_csv(save_path / "per_image_metrics.csv", rows)

    summary: dict[str, object] = {
        "num_samples": n,
        "avg_psnr": total_psnr / max(1, n),
        "avg_ssim": total_ssim / max(1, n),
    }
    if n_lpips:
        summary["avg_lpips"] = total_lpips / n_lpips

    with open(save_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate FaDeNet-style models.")
    p.add_argument("--config", type=str, default="configs/FaDeNet/prism_config.yaml")
    p.add_argument("--model_pkg", type=str, default="FaDeNet")
    p.add_argument("--load_path", type=str, default="", help="Explicit checkpoint path.")
    p.add_argument(
        "--test_epoch",
        type=str,
        default="",
        help="'best', 'latest'/'auto', or an epoch number. Empty uses TEST.TEST_EPOCH from config.",
    )
    p.add_argument("--test_dataset", type=str, default="", help="Override TEST_DATASET from config.")
    p.add_argument("--save_dir", type=str, default="", help="Override result directory.")
    p.add_argument("--save_img", type=str, default="", help="Save predictions with this extension, e.g. png or jpg.")
    p.add_argument("--save_triplet", action="store_true", help="Save input|output|gt triplets.")
    p.add_argument("--max_samples", type=int, default=0, help="Evaluate only the first N samples for smoke tests.")
    p.add_argument("--workers", type=int, default=-1, help="Override DataLoader workers.")
    p.add_argument("--device", type=str, default="", help="Device string, e.g. cuda:0 or cpu.")
    p.add_argument("--no_lpips", action="store_true", help="Skip LPIPS and report only PSNR/SSIM.")
    args_cli = p.parse_args()

    args = load_args(["--config", args_cli.config], default_config=args_cli.config)
    if args_cli.test_dataset:
        args.TEST_DATASET = args_cli.test_dataset
    if args_cli.workers >= 0:
        args.WORKER = args_cli.workers
    args.BATCH_SIZE = 1

    device = init(args, device_arg=args_cli.device)
    model = load_model_from_pkg(args, model_pkg=args_cli.model_pkg).to(device)

    load_path, result_name = resolve_checkpoint(args, load_path=args_cli.load_path, test_epoch=args_cli.test_epoch)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    state = load_state_dict_any(load_path)
    model.load_state_dict(state, strict=True)

    save_path = Path(args_cli.save_dir) if args_cli.save_dir else Path(args.TEST_RESULT_DIR) / result_name
    save_path.mkdir(parents=True, exist_ok=True)
    log_path = save_path / "result.log"
    set_logging(str(log_path))

    logging.warning(datetime.now())
    logging.warning("model package: %s" % args_cli.model_pkg)
    logging.warning("config: %s" % args_cli.config)
    logging.warning("test dataset: %s" % args.TEST_DATASET)
    logging.warning("load model from %s" % load_path)
    logging.warning("save results to %s" % save_path)
    logging.warning("device: %s" % device)

    metrics = Metrics(device, use_lpips=not args_cli.no_lpips)
    test_loader = create_dataset(args, data_path=args.TEST_DATASET, mode="test")
    summary = test_loop(
        args,
        test_loader,
        model,
        metrics,
        save_path=save_path,
        save_img=args_cli.save_img,
        save_triplet=args_cli.save_triplet,
        max_samples=int(args_cli.max_samples),
    )

    msg = "Avg. PSNR %.4f, SSIM %.4f" % (float(summary["avg_psnr"]), float(summary["avg_ssim"]))
    if "avg_lpips" in summary:
        msg += ", LPIPS %.4f" % float(summary["avg_lpips"])
    logging.warning(msg)


if __name__ == "__main__":
    main()
