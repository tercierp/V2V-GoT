import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class OSMViTEncoder(nn.Module):
    """
    Frozen ViT encoder for OSM map images (256x256 → 64 spatial tokens).
    Default: timm/vit_large_patch16_dinov3.sat493m (satellite-pretrained).
    Input images must be pre-normalized and resized to 224x224.
    """

    def __init__(self, model_name='timm/vit_large_patch16_dinov3.sat493m', num_output_tokens=64):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.num_output_tokens = num_output_tokens  # must be a perfect square (e.g. 64 = 8×8)

        # TimmWrapperConfig does not expose hidden_size directly — try multiple paths
        if hasattr(self.model.config, 'hidden_size'):
            self._hidden_size = self.model.config.hidden_size
        elif hasattr(self.model, 'timm_model') and hasattr(self.model.timm_model, 'num_features'):
            self._hidden_size = self.model.timm_model.num_features
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.model(pixel_values=dummy)
                self._hidden_size = out.last_hidden_state.shape[-1]

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, images):
        # images: [B, 3, 224, 224]
        outputs = self.model(
            pixel_values=images.to(device=self.device, dtype=self.dtype)
        )
        # skip CLS + register tokens (timm DINOv3 has 4 register tokens)
        # num_prefix_tokens = 1 (CLS) + N registers
        num_prefix = getattr(getattr(self.model, 'timm_model', None), 'num_prefix_tokens', 1)
        patch_tokens = outputs.last_hidden_state[:, num_prefix:, :]

        B, N, C = patch_tokens.shape
        grid_size = int(N ** 0.5)                       # e.g. 16 for 224×224 patch16
        out_grid  = int(self.num_output_tokens ** 0.5)  # e.g. 8 for 64 tokens

        # [B, C, grid, grid] → adaptive pool → [B, C, out_grid, out_grid]
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)
        x = F.adaptive_avg_pool2d(x, (out_grid, out_grid))
        # → [B, num_output_tokens, C]
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        return x
