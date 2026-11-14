from __future__ import annotations

import torch

from .common import PSNR, SSIM


class create_metrics:
    """PRISM paired metrics: LPIPS, PSNR, and SSIM."""

    def __init__(self, args, device: torch.device):
        if args.DATA_TYPE != "PRISM":
            raise ValueError(f"This release supports DATA_TYPE='PRISM' only, got {args.DATA_TYPE!r}.")

        import lpips

        self.lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        self.fast_ssim = SSIM().to(device).eval()
        self.fast_psnr = PSNR().to(device).eval()

    @torch.no_grad()
    def compute(self, out_img: torch.Tensor, gt: torch.Tensor) -> tuple[float, float, float]:
        pred = torch.clamp(out_img, min=0, max=1)
        target = torch.clamp(gt, min=0, max=1)

        res_psnr = float(self.fast_psnr(pred, target).item())
        res_ssim = float(self.fast_ssim(pred, target).item())
        res_lpips = float(self.lpips_fn.forward(pred, target, normalize=True).item())
        return res_lpips, res_psnr, res_ssim
