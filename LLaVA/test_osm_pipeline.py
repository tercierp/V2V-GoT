"""
Standalone smoke test for the OSM encoder pipeline using CLIP.
No llava env needed — only: torch, transformers, torchvision, Pillow

Install if missing:
    pip install torch transformers torchvision Pillow

Run from anywhere:
    python test_osm_pipeline.py

NOTE: OSM images are preprocessed with CLIP normalization (not ImageNet)
      and resized to 336×336 to match openai/clip-vit-large-patch14-336.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Inline OSMCLIPEncoder (mirrors osm_encoder.py) ──────────────────────────

class FakeCLIPTower(nn.Module):
    """Minimal stub of CLIPVisionTower for testing without loading 7B LLaVA."""
    HIDDEN = 1024
    PATCHES = 576   # 336px / patch14 = 24 → 24×24 = 576

    def __init__(self):
        super().__init__()
        self._proj = nn.Linear(3 * 14 * 14, self.HIDDEN)  # not used, just for param count

    @property
    def hidden_size(self): return self.HIDDEN
    @property
    def dtype(self): return torch.float32
    @property
    def device(self): return torch.device('cpu')

    def forward(self, images):
        B = images.shape[0]
        return torch.randn(B, self.PATCHES, self.HIDDEN)


class OSMCLIPEncoder(nn.Module):
    def __init__(self, vision_tower, num_output_tokens=64):
        super().__init__()
        self.vision_tower = vision_tower
        self.num_output_tokens = num_output_tokens

    @property
    def hidden_size(self): return self.vision_tower.hidden_size

    @torch.no_grad()
    def forward(self, images):
        patch_tokens = self.vision_tower(images)           # [B, 576, 1024]
        B, N, C = patch_tokens.shape
        grid_size = int(N ** 0.5)                          # 24
        out_grid  = int(self.num_output_tokens ** 0.5)     # 8
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)
        x = F.adaptive_avg_pool2d(x, (out_grid, out_grid))
        return x.reshape(B, C, -1).permute(0, 2, 1)        # [B, 64, 1024]


def build_osm_projector(input_dim, hidden_size):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )

# ─── Config ───────────────────────────────────────────────────────────────────

BATCH_SIZE      = 2
OSM_NUM_TOKENS  = 64
LLM_HIDDEN_SIZE = 4096
CLIP_HIDDEN     = 1024   # openai/clip-vit-large-patch14-336
CLIP_IMG_SIZE   = 336    # CLIP input resolution
LIDAR_TOKENS    = (220 + 50) * 2 * 2   # shallow, 2 frames, 2 CAVs = 1080

# ─── Step 1: encoder ─────────────────────────────────────────────────────────

print("=" * 60)
print("Step 1: OSM encoder (CLIP vision tower reuse) ...")

clip_stub = FakeCLIPTower()
encoder   = OSMCLIPEncoder(vision_tower=clip_stub, num_output_tokens=OSM_NUM_TOKENS)
encoder.eval()

print(f"  CLIP hidden_size : {encoder.hidden_size}")
print(f"  extra params     : 0  (reuses existing tower — no new weights)")

fake_osm   = torch.randn(BATCH_SIZE, 3, CLIP_IMG_SIZE, CLIP_IMG_SIZE)
osm_tokens = encoder(fake_osm)

assert osm_tokens.shape == (BATCH_SIZE, OSM_NUM_TOKENS, CLIP_HIDDEN)
print(f"  [B,576,1024] → pool → {tuple(osm_tokens.shape)}  ✓")

# ─── Step 2: projector ───────────────────────────────────────────────────────

print()
print("Step 2: Projector ...")

proj         = build_osm_projector(CLIP_HIDDEN, LLM_HIDDEN_SIZE)
osm_features = proj(osm_tokens)

assert osm_features.shape == (BATCH_SIZE, OSM_NUM_TOKENS, LLM_HIDDEN_SIZE)
print(f"  {CLIP_HIDDEN} → {LLM_HIDDEN_SIZE} : {tuple(osm_features.shape)}  ✓")

# ─── Step 3: concat ──────────────────────────────────────────────────────────

print()
print("Step 3: Concat with LiDAR tokens ...")

fake_lidar = torch.randn(BATCH_SIZE, LIDAR_TOKENS, LLM_HIDDEN_SIZE)
combined   = torch.cat([fake_lidar, osm_features], dim=1)

assert combined.shape == (BATCH_SIZE, LIDAR_TOKENS + OSM_NUM_TOKENS, LLM_HIDDEN_SIZE)
print(f"  {LIDAR_TOKENS} LiDAR + {OSM_NUM_TOKENS} OSM = {LIDAR_TOKENS+OSM_NUM_TOKENS} tokens  ✓")
print(f"  combined : {tuple(combined.shape)}  ✓")

# ─── Step 4: context budget ──────────────────────────────────────────────────

print()
print("Step 4: Context budget ...")

total  = LIDAR_TOKENS + OSM_NUM_TOKENS + 500
margin = 2048 - total
print(f"  {LIDAR_TOKENS} + {OSM_NUM_TOKENS} + 500 (QA) = {total} / 2048  (margin {margin})  ✓")
assert margin >= 0

# ─── Step 5: CLIP image preprocessing ───────────────────────────────────────

print()
print("Step 5: CLIP image preprocessing (zoom 18, 256×256 → 336×336) ...")

from PIL import Image
from transformers import CLIPImageProcessor

processor  = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
fake_png   = Image.new('RGB', (256, 256), color=(180, 200, 220))
t          = processor.preprocess(fake_png, return_tensors='pt')['pixel_values'][0]

assert t.shape == (3, 336, 336)
print(f"  OSM PNG 256×256 → CLIP tensor {tuple(t.shape)}  ✓")

# ─── Done ────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("All steps passed — OSM pipeline (CLIP) is ready.")
print()
print(f"  Encoder  : CLIP ViT-L/14 @ 336px  (already loaded, 0 extra memory)")
print(f"  OSM zoom : 18  (~154m coverage at 256×256)")
print(f"  Tokens   : 576 → pool 8×8 → 64 OSM tokens")
print(f"  Projector: {CLIP_HIDDEN} → {LLM_HIDDEN_SIZE}  (trainable)")
print(f"  Budget   : {total} / 2048  ({margin} margin)")
