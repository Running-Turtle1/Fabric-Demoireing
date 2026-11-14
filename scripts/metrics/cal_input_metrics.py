from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from configs.common.config import load_args as load_config_args
from dataloader.paired_image.load_data import create_dataset
from utils.metric import create_metrics


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/LPIPS of input images against GT (paired datasets like PRISM)."
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/FaDeNet/prism_config.yaml",
        help="FaDeNet YAML config (must set DATA_TYPE=PRISM).",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Override dataset root (e.g. dataset/PRISM/test). If empty, uses TEST_DATASET from config.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to evaluate. Only affects the default --data_root selection and dataloader mode.",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Where to write per-sample metrics CSV. Default: outputs/input_metrics_<split>.csv",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override DataLoader num_workers. Use 0 if multiprocessing is blocked in your environment.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate first N samples (0 = full split).",
    )
    args_cli = p.parse_args()

    args = load_config_args(["--config", args_cli.config])
    if args.DATA_TYPE != "PRISM":
        raise ValueError(f"DATA_TYPE must be PRISM for this script (got {args.DATA_TYPE!r}).")

    args.BATCH_SIZE = 1
    if args_cli.workers is not None:
        args.WORKER = int(args_cli.workers)

    split = str(args_cli.split)
    if args_cli.data_root:
        data_root = Path(args_cli.data_root)
    else:
        data_root = Path(args.TRAIN_DATASET if split == "train" else args.TEST_DATASET)

    out_csv = Path(args_cli.out_csv) if args_cli.out_csv else Path("outputs") / f"input_metrics_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = create_metrics(args, device=device)

    loader = create_dataset(args, data_path=str(data_root), mode=split)

    rows: list[dict[str, object]] = []
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    n = 0

    for batch in tqdm(loader, desc=f"{split} input metrics"):
        in_img = batch["in_img"].to(device)
        label = batch["label"].to(device)
        number = batch["number"][0] if isinstance(batch["number"], (list, tuple)) else str(batch["number"])

        cur_lpips, cur_psnr, cur_ssim = metric.compute(in_img, label)
        psnr_sum += float(cur_psnr)
        ssim_sum += float(cur_ssim)
        lpips_sum += float(cur_lpips)
        n += 1

        rows.append(
            {
                "number": number,
                "psnr": float(cur_psnr),
                "ssim": float(cur_ssim),
                "lpips": float(cur_lpips),
            }
        )

        if int(args_cli.limit) > 0 and n >= int(args_cli.limit):
            break

    denom = max(1, n)
    avg_psnr = psnr_sum / denom
    avg_ssim = ssim_sum / denom
    avg_lpips = lpips_sum / denom

    rows.append({"number": "__mean__", "psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["number", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] n={n} mean_psnr={avg_psnr:.4f} mean_ssim={avg_ssim:.4f} mean_lpips={avg_lpips:.4f}")
    print(f"[OK] Wrote: {str(out_csv)}")


if __name__ == "__main__":
    main()
