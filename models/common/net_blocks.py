from __future__ import annotations

"""Shared FaDeNet network blocks."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    if multiple <= 1:
        _, _, h, w = x.shape
        return x, (0, 0, int(h), int(w))

    _, _, h, w = x.shape
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)
    pad_h = new_h - h
    pad_w = new_w - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, int(h), int(w))

    mode = "reflect"
    if h <= 1 or w <= 1 or pad_top >= h or pad_bottom >= h or pad_left >= w or pad_right >= w:
        mode = "replicate"
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    return x_pad, (int(pad_top), int(pad_left), int(h), int(w))


def crop_to_original(x: torch.Tensor, crop: tuple[int, int, int, int]) -> torch.Tensor:
    pad_top, pad_left, h, w = crop
    return x[:, :, pad_top : pad_top + h, pad_left : pad_left + w]


def _gaussian_2d_kernel(kernel_size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError(f"kernel_size must be odd and >1, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be >0, got {sigma}")

    k = kernel_size
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g1 = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    g2 = g2 / g2.sum()
    return g2


class FixedGaussianBlur(nn.Module):
    def __init__(self, channels: int = 3, kernel_size: int = 5, sigma: float = 1.2):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        self.register_buffer("_kernel", torch.empty(0), persistent=False)

    def _get_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._kernel.numel() == 0 or self._kernel.device != device or self._kernel.dtype != dtype:
            k2 = _gaussian_2d_kernel(self.kernel_size, self.sigma, device=device, dtype=dtype)
            self._kernel = k2.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        return self._kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != self.channels:
            raise ValueError(f"Expected NCHW with C={self.channels}, got shape={tuple(x.shape)!r}")

        k = self.kernel_size
        pad = k // 2
        b, c, h, w = x.shape

        mode = "reflect"
        if h <= 1 or w <= 1 or pad >= h or pad >= w:
            mode = "replicate"
        x_pad = F.pad(x, (pad, pad, pad, pad), mode=mode)
        weight = self._get_weight(device=x.device, dtype=x.dtype)
        return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=self.channels)


class PixelUnshuffleDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.unshuffle(x))


class PixelShuffleUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, bias=True)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


class SAGBBlock(nn.Module):
    """Spectral-anisotropic gated block."""

    def __init__(self, channels: int, *, use_fft_gate: bool):
        super().__init__()
        ch = int(channels)
        self.use_fft_gate = bool(use_fft_gate)

        self.norm = nn.GroupNorm(1, ch)

        self.dw3 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        self.dw_dil = nn.Conv2d(ch, ch, kernel_size=3, padding=2, dilation=2, groups=ch, bias=False)
        self.dw_h = nn.Conv2d(ch, ch, kernel_size=(1, 5), padding=(0, 2), groups=ch, bias=False)
        self.dw_v = nn.Conv2d(ch, ch, kernel_size=(5, 1), padding=(2, 0), groups=ch, bias=False)

        self.spatial_fuse = nn.Conv2d(ch * 4, ch, kernel_size=1, bias=True)

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

        fs = torch.cat([self.dw3(x), self.dw_dil(x), self.dw_h(x), self.dw_v(x)], dim=1)
        fs = self.spatial_fuse(F.gelu(fs))

        if self.use_fft_gate:
            mag = torch.fft.rfft2(x, norm="ortho").abs()
            mag_stats = mag.mean(dim=(-2, -1), keepdim=True)
            w = self.freq_mlp(mag_stats)  # type: ignore[misc]
            ff = x * w
        else:
            ff = x

        cat = torch.cat([fs, ff], dim=1)
        g = torch.sigmoid(self.gate(cat))
        upd = self.phi(cat)
        return x0 + g * upd


class TrunkOut(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.res = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1, bias=True)
        self.mask = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        res = torch.tanh(self.res(x))
        mask = torch.sigmoid(self.mask(x))
        return res, mask


class DetailGuardHead(nn.Module):
    def __init__(self, *, alpha: float):
        super().__init__()
        self.alpha = float(alpha)
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, detail: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        rd = torch.tanh(self.net(torch.cat([detail, mask], dim=1)))
        return detail - self.alpha * mask * rd


@dataclass(frozen=True)
class FaDeNetAux:
    base: torch.Tensor
    detail: torch.Tensor
    base_hat: torch.Tensor
    detail_hat: torch.Tensor
    mask: torch.Tensor
    res_base: torch.Tensor


class FaDeNet(nn.Module):
    """FaDeNet demoire network."""

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
    ):
        super().__init__()
        self.pad_multiple = int(pad_multiple)

        ch0 = int(base_channels)
        ch1 = ch0 * 2
        ch2 = ch0 * 4
        ch3 = ch0 * 8

        if len(num_blocks) != 4:
            raise ValueError(f"num_blocks must have 4 ints (for 4 scales), got {num_blocks!r}")
        if len(fft_gate_scales) != 4:
            raise ValueError(f"fft_gate_scales must have 4 bools (for 4 scales), got {fft_gate_scales!r}")

        self.blur = FixedGaussianBlur(channels=3, kernel_size=blur_kernel_size, sigma=blur_sigma)

        self.stem = nn.Conv2d(3, ch0, kernel_size=3, padding=1, bias=True)

        self.enc0 = nn.Sequential(*[SAGBBlock(ch0, use_fft_gate=fft_gate_scales[0]) for _ in range(int(num_blocks[0]))])
        self.down0 = PixelUnshuffleDown(ch0, ch1)

        self.enc1 = nn.Sequential(*[SAGBBlock(ch1, use_fft_gate=fft_gate_scales[1]) for _ in range(int(num_blocks[1]))])
        self.down1 = PixelUnshuffleDown(ch1, ch2)

        self.enc2 = nn.Sequential(*[SAGBBlock(ch2, use_fft_gate=fft_gate_scales[2]) for _ in range(int(num_blocks[2]))])
        self.down2 = PixelUnshuffleDown(ch2, ch3)

        self.enc3 = nn.Sequential(*[SAGBBlock(ch3, use_fft_gate=fft_gate_scales[3]) for _ in range(int(num_blocks[3]))])

        self.up2 = PixelShuffleUp(ch3, ch2)
        self.fuse2 = nn.Conv2d(ch2 * 2, ch2, kernel_size=1, bias=True)
        self.dec2 = nn.Sequential(*[SAGBBlock(ch2, use_fft_gate=fft_gate_scales[2]) for _ in range(int(num_blocks[2]))])

        self.up1 = PixelShuffleUp(ch2, ch1)
        self.fuse1 = nn.Conv2d(ch1 * 2, ch1, kernel_size=1, bias=True)
        self.dec1 = nn.Sequential(*[SAGBBlock(ch1, use_fft_gate=fft_gate_scales[1]) for _ in range(int(num_blocks[1]))])

        self.up0 = PixelShuffleUp(ch1, ch0)
        self.fuse0 = nn.Conv2d(ch0 * 2, ch0, kernel_size=1, bias=True)
        self.dec0 = nn.Sequential(*[SAGBBlock(ch0, use_fft_gate=fft_gate_scales[0]) for _ in range(int(num_blocks[0]))])

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
