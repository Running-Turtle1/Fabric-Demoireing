from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from configs.common.config import load_args


def _load_state_dict_any(weights_path: str | Path) -> dict[str, torch.Tensor]:
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


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB tensor, got shape={tuple(x.shape)!r}")
    x = x.detach().clamp(0, 1).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x, mode="RGB")


def _pil_to_tensor(path: Path) -> torch.Tensor:
    tr = transforms.ToTensor()
    with Image.open(path) as im:
        return tr(im.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def _infer_one(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    out = model(x)
    return out.clamp(0, 1).cpu()


def _make_collage(rows: list[Image.Image], *, out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to compose.")
    w, h = rows[0].size
    if any(r.size != (w, h) for r in rows):
        raise ValueError("All row images must have the same size.")
    canvas = Image.new("RGB", (w, h * len(rows)))
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Infer FaDeNet on a paired (input, gt) folder and save input|out|gt collage.")
    p.add_argument("--weights", type=str, required=True, help="Checkpoint path (expects {'state_dict': ...} or raw state_dict).")
    p.add_argument(
        "--config",
        type=str,
        default="configs/FaDeNet/prism_config.yaml",
        help="Config YAML used to instantiate the model (must match the checkpoint).",
    )
    p.add_argument(
        "--model_pkg",
        type=str,
        default="FaDeNet",
        help="Model package under models/.",
    )
    p.add_argument("--input_dir", type=str, required=True, help="Input (moire) image folder.")
    p.add_argument("--gt_dir", type=str, required=True, help="GT image folder (same filenames as input_dir).")
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Where to write predicted images (default: <input_dir>/../fadenet_out_<ckpt_stem>).",
    )
    p.add_argument(
        "--collage_path",
        type=str,
        default="",
        help="Where to write the final collage PNG (default: <out_dir>/collage_input_out_gt.png).",
    )
    p.add_argument("--device", type=str, default="", help="Device string, e.g. 'cuda' or 'cpu' (default: auto).")
    args_cli = p.parse_args()

    input_dir = Path(args_cli.input_dir)
    gt_dir = Path(args_cli.gt_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")

    weights_path = Path(args_cli.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    ckpt_stem = weights_path.stem.replace(".", "_")
    out_dir = Path(args_cli.out_dir) if args_cli.out_dir else (input_dir.parent / f"fadenet_out_{ckpt_stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    collage_path = Path(args_cli.collage_path) if args_cli.collage_path else (out_dir / "collage_input_out_gt.png")

    cfg = load_args(["--config", args_cli.config])
    device = torch.device(args_cli.device) if args_cli.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = importlib.import_module(f"models.{args_cli.model_pkg}.net")
    cls = getattr(mod, "FaDeNet", None)
    if cls is None:
        raise AttributeError(f"models.{args_cli.model_pkg}.net must define `FaDeNet`.")

    model = cls(
        base_channels=int(getattr(cfg, "BASE_CHANNELS", 48)),
        num_blocks=tuple(int(x) for x in getattr(cfg, "NUM_BLOCKS", [2, 2, 2, 2])),
        pad_multiple=int(getattr(cfg, "PAD_MULTIPLE", 8)),
        blur_kernel_size=int(getattr(cfg, "BLUR_KERNEL_SIZE", 5)),
        blur_sigma=float(getattr(cfg, "BLUR_SIGMA", 1.2)),
        detail_alpha=float(getattr(cfg, "DETAIL_ALPHA", 0.35)),
        fft_gate_scales=tuple(bool(x) for x in getattr(cfg, "FFT_GATE_SCALES", [False, False, True, True])),
    ).to(device)
    model.eval()

    state = _load_state_dict_any(weights_path)
    model.load_state_dict(state, strict=True)

    gt_paths = sorted(p for p in gt_dir.glob("*.png"))
    if not gt_paths:
        raise FileNotFoundError(f"No .png found in gt_dir: {gt_dir}")

    rows: list[Image.Image] = []
    for gt_path in gt_paths:
        name = gt_path.name
        in_path = input_dir / name
        if not in_path.exists():
            raise FileNotFoundError(f"Missing input image for {name}: {in_path}")

        in_t = _pil_to_tensor(in_path)
        gt_t = _pil_to_tensor(gt_path)
        out_t = _infer_one(model, in_t, device)

        out_path = out_dir / name
        _tensor_to_pil(out_t).save(out_path)

        in_im = Image.open(in_path).convert("RGB")
        out_im = _tensor_to_pil(out_t)
        gt_im = Image.open(gt_path).convert("RGB")
        if in_im.size != out_im.size or in_im.size != gt_im.size:
            raise ValueError(f"Size mismatch for {name}: in={in_im.size}, out={out_im.size}, gt={gt_im.size}")

        w, h = in_im.size
        row = Image.new("RGB", (w * 3, h))
        row.paste(in_im, (0, 0))
        row.paste(out_im, (w, 0))
        row.paste(gt_im, (w * 2, 0))
        rows.append(row)

    _make_collage(rows, out_path=collage_path)
    print(f"[OK] Wrote outputs to: {out_dir}")
    print(f"[OK] Wrote collage to: {collage_path}")


if __name__ == "__main__":
    main()
