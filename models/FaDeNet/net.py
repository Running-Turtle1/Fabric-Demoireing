from __future__ import annotations

"""FaDeNet model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from models.common.net_blocks import (
    DetailGuardHead,
    PixelShuffleUp,
    PixelUnshuffleDown,
    TrunkOut,
    FaDeNetAux,
    crop_to_original,
    pad_to_multiple,
)
from models.common.components import AdaptiveDWGaussianBlur


class SAGBProBlock(nn.Module):
    """SAGBPro block with spatial branches and optional FFT gating."""

    def __init__(self, channels: int, *, use_fft_gate: bool = True):
        super().__init__()
        ch = int(channels)
        self.use_fft_gate = bool(use_fft_gate)

        self.norm = nn.GroupNorm(1, ch)
        self.expand = nn.Conv2d(ch, ch * 4, kernel_size=1, bias=True)

        branch_ch = ch
        self.branch1 = nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, groups=branch_ch, bias=False)
        self.branch2 = nn.Conv2d(branch_ch, branch_ch, kernel_size=(1, 5), padding=(0, 2), groups=branch_ch, bias=False)
        self.branch3 = nn.Conv2d(branch_ch, branch_ch, kernel_size=(5, 1), padding=(2, 0), groups=branch_ch, bias=False)
        self.branch4 = nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=2, dilation=2, groups=branch_ch, bias=False)

        self.project = nn.Conv2d(ch * 4, ch, kernel_size=1, bias=True)

        if self.use_fft_gate:
            hidden = max(8, ch // 4)
            self.freq_mlp = nn.Sequential(
                nn.Conv2d(ch, hidden, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, ch, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.freq_mlp = None

        self.gate = nn.Conv2d(ch * 2, ch, kernel_size=1, bias=True)
        self.phi = nn.Conv2d(ch * 2, ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        x = self.norm(x)
        x = self.expand(x)
        c1, c2, c3, c4 = torch.split(x, x.size(1) // 4, dim=1)

        y1 = F.gelu(self.branch1(c1))
        y2 = F.gelu(self.branch2(c2))
        y3 = F.gelu(self.branch3(c3))
        y4 = F.gelu(self.branch4(c4))
        fs = self.project(torch.cat([y1, y2, y3, y4], dim=1))

        if self.use_fft_gate:
            autocast_ctx = torch.cuda.amp.autocast(enabled=False) if x0.is_cuda else nullcontext()
            with autocast_ctx:
                mag = torch.fft.rfft2(x0.float(), norm="ortho").abs()
                mag_stats = mag.mean(dim=(-2, -1), keepdim=True)
                w = self.freq_mlp(mag_stats)  # type: ignore[misc]
            ff = x0 * w.to(dtype=x0.dtype)
        else:
            ff = x0

        cat = torch.cat([fs, ff], dim=1)
        g = torch.sigmoid(self.gate(cat))
        upd = self.phi(cat)
        return x0 + g * upd


class FaDeNet(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 48,
        num_blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
        pad_multiple: int = 8,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.2,
        detail_alpha: float = 0.35,
        fft_gate_scales: tuple[bool, bool, bool, bool] = (False, False, True, True),
    ) -> None:
        super().__init__()
        self.pad_multiple = int(pad_multiple)

        ch0 = int(base_channels)
        ch1 = ch0 * 2
        ch2 = ch0 * 4
        ch3 = ch0 * 8

        if len(num_blocks) != 4:
            raise ValueError(f"num_blocks must have 4 ints (for 4 scales), got {num_blocks!r}")
        if len(fft_gate_scales) != 4:
            raise ValueError(f"fft_gate_scales must have 4 bools, got {fft_gate_scales!r}")

        self.blur = AdaptiveDWGaussianBlur(channels=3, kernel_size=blur_kernel_size, sigma=blur_sigma)

        self.stem = nn.Conv2d(3, ch0, kernel_size=3, padding=1, bias=True)

        self.enc0 = nn.Sequential(
            *[SAGBProBlock(ch0, use_fft_gate=fft_gate_scales[0]) for _ in range(int(num_blocks[0]))]
        )
        self.down0 = PixelUnshuffleDown(ch0, ch1)

        self.enc1 = nn.Sequential(
            *[SAGBProBlock(ch1, use_fft_gate=fft_gate_scales[1]) for _ in range(int(num_blocks[1]))]
        )
        self.down1 = PixelUnshuffleDown(ch1, ch2)

        self.enc2 = nn.Sequential(
            *[SAGBProBlock(ch2, use_fft_gate=fft_gate_scales[2]) for _ in range(int(num_blocks[2]))]
        )
        self.down2 = PixelUnshuffleDown(ch2, ch3)

        self.enc3 = nn.Sequential(
            *[SAGBProBlock(ch3, use_fft_gate=fft_gate_scales[3]) for _ in range(int(num_blocks[3]))]
        )

        self.up2 = PixelShuffleUp(ch3, ch2)
        self.fuse2 = nn.Conv2d(ch2 * 2, ch2, kernel_size=1, bias=True)
        self.dec2 = nn.Sequential(
            *[SAGBProBlock(ch2, use_fft_gate=fft_gate_scales[2]) for _ in range(int(num_blocks[2]))]
        )

        self.up1 = PixelShuffleUp(ch2, ch1)
        self.fuse1 = nn.Conv2d(ch1 * 2, ch1, kernel_size=1, bias=True)
        self.dec1 = nn.Sequential(
            *[SAGBProBlock(ch1, use_fft_gate=fft_gate_scales[1]) for _ in range(int(num_blocks[1]))]
        )

        self.up0 = PixelShuffleUp(ch1, ch0)
        self.fuse0 = nn.Conv2d(ch0 * 2, ch0, kernel_size=1, bias=True)
        self.dec0 = nn.Sequential(
            *[SAGBProBlock(ch0, use_fft_gate=fft_gate_scales[0]) for _ in range(int(num_blocks[0]))]
        )

        self.trunk_out = TrunkOut(ch0)
        self.detail_head = DetailGuardHead(alpha=detail_alpha)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        if x.ndim != 4 or int(x.shape[1]) != 3:
            raise ValueError(f"Expected NCHW with C=3, got shape={tuple(x.shape)!r}")

        x_pad, crop = pad_to_multiple(x, self.pad_multiple)

        base = self.blur(x_pad)
        detail = x_pad - base

        h0 = self.stem(base)
        s0 = self.enc0(h0)

        h1 = self.down0(s0)
        s1 = self.enc1(h1)

        h2 = self.down1(s1)
        s2 = self.enc2(h2)

        h3 = self.down2(s2)
        h3 = self.enc3(h3)

        d2 = self.up2(h3)
        d2 = self.dec2(self.fuse2(torch.cat([d2, s2], dim=1)))

        d1 = self.up1(d2)
        d1 = self.dec1(self.fuse1(torch.cat([d1, s1], dim=1)))

        d0 = self.up0(d1)
        d0 = self.dec0(self.fuse0(torch.cat([d0, s0], dim=1)))

        res_base, mask = self.trunk_out(d0)
        base_hat = base - mask * res_base

        detail_hat = self.detail_head(detail, mask)

        out = base_hat + detail_hat
        out = crop_to_original(out, crop)

        if not return_aux:
            return out

        aux = FaDeNetAux(
            base=crop_to_original(base, crop),
            detail=crop_to_original(detail, crop),
            base_hat=crop_to_original(base_hat, crop),
            detail_hat=crop_to_original(detail_hat, crop),
            mask=crop_to_original(mask, crop),
            res_base=crop_to_original(res_base, crop),
        )
        return out, aux
