import torch
import torch.nn as nn
import torch.nn.functional as F


class OSMCLIPEncoder(nn.Module):
    """
    OSM map image encoder that reuses the CLIP vision tower already loaded
    in the LLaVA pipeline — zero extra GPU memory cost.

    The existing CLIP tower (openai/clip-vit-large-patch14-336) produces
    576 patch tokens at 336px. We pool them spatially down to num_output_tokens
    (default 64 = 8×8) before the projector.

    Input images must be pre-processed with CLIPImageProcessor
    (normalize with CLIP mean/std, resize to 336×336).
    """

    def __init__(self, vision_tower, num_output_tokens=64):
        super().__init__()
        # vision_tower is the already-loaded CLIPVisionTower — no new weights
        self.vision_tower = vision_tower
        self.num_output_tokens = num_output_tokens   # must be a perfect square

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @torch.no_grad()
    def forward(self, images):
        # images: [B, 3, 336, 336]  (CLIP preprocessed)
        # vision_tower forward already strips CLS and returns patch tokens
        patch_tokens = self.vision_tower(images)   # [B, 576, 1024]

        B, N, C = patch_tokens.shape
        grid_size = int(N ** 0.5)                       # 24 for CLIP 336px patch14
        out_grid  = int(self.num_output_tokens ** 0.5)  # 8 for 64 tokens

        # spatial pool: [B, C, 24, 24] → [B, C, 8, 8] → [B, 64, C]
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)
        x = F.adaptive_avg_pool2d(x, (out_grid, out_grid))
        return x.reshape(B, C, -1).permute(0, 2, 1)
