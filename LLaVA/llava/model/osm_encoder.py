"""
osm_encoder.py
==============

Small CNN encoder that maps an OSM BEV tensor (4 channels, 200x200, 0.5 m/cell)
to a fixed sequence of N tokens, each of dimension `hidden_size` (matches the
LLM's embedding dimension, e.g. 4096 for Vicuna 7B).

Architecture:
    input:  (B, 4, 200, 200)        4 channels: drivable, road_class,
                                     oneway_dir, signed_dist (norm)

    stem:   3x3 conv,  4 → 32, stride 1   --> (B, 32, 200, 200)
            ReLU
            3x3 conv, 32 → 64, stride 2   --> (B, 64, 100, 100)
            ReLU
            3x3 conv, 64 → 128, stride 2  --> (B, 128, 50, 50)
            ReLU
            3x3 conv, 128 → 256, stride 2 --> (B, 256, 25, 25)
            ReLU

    pool:   AdaptiveAvgPool2d(2)          --> (B, 256, 2, 2) = 4 spatial cells
                                              encoding TL/TR/BL/BR quadrants

    proj:   Linear 256 → hidden_size      --> (B, 4, hidden_size)

Output shape: (B, 4, hidden_size)
The four tokens correspond to four spatial quadrants of the BEV, indexed
(roughly) as: [front-left, front-right, back-left, back-right] in body frame.
This gives the LLM "road on the left", "road on the right" semantics.

Parameter count: ~600K (encoder) + 1M (projector) ≈ 1.6M params.
Trains from scratch alongside the LoRA adapters during fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OSMEncoder(nn.Module):
    """OSM BEV → N tokens of dim `hidden_size`.

    Args:
        in_channels: number of channels in the input BEV (default 4).
        hidden_size: embedding dimension of the target LLM (4096 for Vicuna 7B).
        num_tokens: how many output tokens to produce. Must be a perfect square
            so we can lay them out as a sqrt(N) x sqrt(N) grid via pooling.
            Default 4 → 2x2 grid of quadrant tokens.
        cnn_widths: tuple of channel widths for the 4 conv stages.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 4096,
        num_tokens: int = 4,
        cnn_widths: tuple[int, int, int, int] = (32, 64, 128, 256),
    ):
        super().__init__()

        # num_tokens must be a perfect square (we lay them out spatially)
        token_grid = int(num_tokens ** 0.5)
        assert token_grid * token_grid == num_tokens, (
            f"num_tokens={num_tokens} is not a perfect square"
        )
        self.num_tokens = num_tokens
        self.token_grid = token_grid
        self.hidden_size = hidden_size

        c1, c2, c3, c4 = cnn_widths

        # CNN backbone — 4 conv stages with stride 2 progressive downsampling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Pool to a (token_grid x token_grid) spatial map
        self.pool = nn.AdaptiveAvgPool2d(token_grid)

        # Project pooled features to LLM embedding dimension
        self.proj = nn.Linear(c4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) OSM BEV tensor in body frame.
               H and W can be anything >= 16 (we pool to fixed token grid).
        Returns:
            tokens: (B, num_tokens, hidden_size) ready to concatenate with
                    LLaVA's scene/object/text tokens.
        """
        # CNN: (B, in_C, H, W) -> (B, c4, H/8, W/8)
        feats = self.conv(x)
        # Pool: (B, c4, H/8, W/8) -> (B, c4, token_grid, token_grid)
        pooled = self.pool(feats)
        # Reshape into a token sequence: (B, c4, token_grid*token_grid)
        b, c, h, w = pooled.shape
        pooled = pooled.view(b, c, h * w).transpose(1, 2)  # (B, num_tokens, c4)
        # Project to LLM embedding dim
        tokens = self.proj(pooled)  # (B, num_tokens, hidden_size)
        return tokens

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    encoder = OSMEncoder(
        in_channels=4,
        hidden_size=4096,
        num_tokens=4,
    )
    print(f"OSMEncoder params: {encoder.num_parameters:,}")

    # Single sample
    x = torch.randn(1, 4, 200, 200)
    tokens = encoder(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(tokens.shape)}")
    assert tokens.shape == (1, 4, 4096), f"Unexpected shape {tokens.shape}"

    # Batched
    x = torch.randn(8, 4, 200, 200)
    tokens = encoder(x)
    assert tokens.shape == (8, 4, 4096), f"Unexpected shape {tokens.shape}"
    print(f"Batched check OK: {tuple(x.shape)} -> {tuple(tokens.shape)}")

    # Gradient sanity: a fake loss should produce non-zero gradients on every param
    loss = tokens.pow(2).mean()
    loss.backward()
    n_with_grad = sum(1 for p in encoder.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in encoder.parameters())
    print(f"Backward pass: {n_with_grad}/{n_total} parameter tensors received gradients")
    assert n_with_grad == n_total, "Some parameters did not receive gradients!"

    print("\nAll smoke tests passed.")
