from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.net_blocks import FaDeNet as _BaseFaDeNet


def _gaussian_2d_kernel(kernel_size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError(f"kernel_size must be odd and >1, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be >0, got {sigma}")

    k = int(kernel_size)
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g1 = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    g2 = g2 / g2.sum()
    return g2


def _pad_like_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = int(kernel_size) // 2
    if pad <= 0:
        return x
    _, _, h, w = x.shape
    mode = "reflect"
    if h <= 1 or w <= 1 or pad >= h or pad >= w:
        mode = "replicate"
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


class LearnableDWGaussianBlur(nn.Module):
    """
    Depthwise learnable blur initialized with a 2D Gaussian kernel.
    """

    def __init__(self, channels: int = 3, kernel_size: int = 5, sigma: float = 1.2) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        self.conv = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            groups=self.channels,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            k2 = _gaussian_2d_kernel(
                self.kernel_size,
                self.sigma,
                device=self.conv.weight.device,
                dtype=self.conv.weight.dtype,
            )
            weight = k2.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
            self.conv.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != self.channels:
            raise ValueError(f"Expected NCHW with C={self.channels}, got shape={tuple(x.shape)!r}")
        x_pad = _pad_like_blur(x, self.kernel_size)
        return self.conv(x_pad)


class AdaptiveDWGaussianBlur(nn.Module):
    """
    Content-adaptive blur:
      base = W * DWConv(x) + (1 - W) * x
    where W is predicted from a lightweight conv branch.
    """

    def __init__(
        self,
        channels: int = 3,
        kernel_size: int = 5,
        sigma: float = 1.2,
        gate_channels: int = 8,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        self.dw_blur = LearnableDWGaussianBlur(
            channels=self.channels,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
        )
        mid = max(4, int(gate_channels))
        self.gate = nn.Sequential(
            nn.Conv2d(self.channels, mid, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != self.channels:
            raise ValueError(f"Expected NCHW with C={self.channels}, got shape={tuple(x.shape)!r}")
        w = self.gate(x)
        b = self.dw_blur(x)
        return w * b + (1.0 - w) * x


class FaDeNetS1(_BaseFaDeNet):
    """
    FaDeNet with learnable depthwise Gaussian blur for base/detail split.
    """

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
        super().__init__(
            base_channels=base_channels,
            num_blocks=num_blocks,
            pad_multiple=pad_multiple,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            detail_alpha=detail_alpha,
            fft_gate_scales=fft_gate_scales,
        )
        self.blur = LearnableDWGaussianBlur(channels=3, kernel_size=blur_kernel_size, sigma=blur_sigma)


class FaDeNetS2(_BaseFaDeNet):
    """
    FaDeNet with content-adaptive base/detail split.
    """

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
        super().__init__(
            base_channels=base_channels,
            num_blocks=num_blocks,
            pad_multiple=pad_multiple,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            detail_alpha=detail_alpha,
            fft_gate_scales=fft_gate_scales,
        )
        self.blur = AdaptiveDWGaussianBlur(channels=3, kernel_size=blur_kernel_size, sigma=blur_sigma)
