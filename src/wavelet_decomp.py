"""
wavelet_decomp.py
=================
Parameter-free, differentiable Haar wavelet functions.

haar_dwt2d(x)  -> LL, LH, HL, HH
haar_idwt2d(LL, LH, HL, HH) -> x
"""

import torch


def haar_dwt2d(x):
    """
    Single-level 2-D Haar discrete wavelet transform.

    Args:
        x: [B, C, H, W]  -- H and W must be even
    Returns:
        LL: [B, C, H/2, W/2]  low-pass  (mean)
        LH: [B, C, H/2, W/2]  horizontal detail
        HL: [B, C, H/2, W/2]  vertical detail
        HH: [B, C, H/2, W/2]  diagonal detail
    Note:
        LL is in [0, 1] when x is in [0, 1].
        LH, HL, HH are in [-0.5, 0.5] when x is in [0, 1].
    """
    assert x.shape[-1] % 2 == 0 and x.shape[-2] % 2 == 0, \
        f"H and W must be even, got {x.shape[-2]}x{x.shape[-1]}"

    p00 = x[:, :, 0::2, 0::2]
    p01 = x[:, :, 0::2, 1::2]
    p10 = x[:, :, 1::2, 0::2]
    p11 = x[:, :, 1::2, 1::2]

    LL = (p00 + p01 + p10 + p11) * 0.25
    LH = (p00 + p01 - p10 - p11) * 0.25
    HL = (p00 - p01 + p10 - p11) * 0.25
    HH = (p00 - p01 - p10 + p11) * 0.25

    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH):
    """
    Single-level 2-D Haar inverse discrete wavelet transform.

    Args:
        LL, LH, HL, HH: each [B, C, H, W]
    Returns:
        x: [B, C, H*2, W*2]
    """
    B, C, H, W = LL.shape
    x = torch.zeros(B, C, H * 2, W * 2, dtype=LL.dtype, device=LL.device)
    x[:, :, 0::2, 0::2] = LL + LH + HL + HH
    x[:, :, 0::2, 1::2] = LL + LH - HL - HH
    x[:, :, 1::2, 0::2] = LL - LH + HL - HH
    x[:, :, 1::2, 1::2] = LL - LH - HL + HH
    return x
