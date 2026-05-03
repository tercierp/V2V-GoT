"""
Standalone smoke test for the OSM encoder pipeline.
No llava env needed — only: torch, transformers, torchvision, Pillow

Install if missing:
    pip install torch transformers torchvision Pillow

Run from anywhere:
    python test_osm_pipeline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Inline OSMViTEncoder (copy of osm_encoder.py) ───────────────────────────

from transformers import AutoModel

class OSMViTEncoder(nn.Module):
    def __init__(self, model_name='timm/vit_large_patch16_dinov3.sat493m', num_output_tokens=64):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.num_output_tokens = num_output_tokens
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

    @torch.no_grad()
    def forward(self, images):
        outputs = self.model(pixel_values=images)
        # skip CLS + register tokens (DINOv3/sat493m has 4 register tokens)
        num_prefix = getattr(getattr(self.model, 'timm_model', None), 'num_prefix_tokens', 1)
        patch_tokens = outputs.last_hidden_state[:, num_prefix:, :]
        B, N, C = patch_tokens.shape
        grid_size = int(N ** 0.5)
        out_grid  = int(self.num_output_tokens ** 0.5)
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)
        x = F.adaptive_avg_pool2d(x, (out_grid, out_grid))
        return x.reshape(B, C, -1).permute(0, 2, 1)          # [B, 64, C]


def build_osm_projector(input_dim, hidden_size):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )

# ─── Config ───────────────────────────────────────────────────────────────────

BATCH_SIZE       = 2
OSM_ENCODER_NAME = 'timm/vit_large_patch16_dinov3.sat493m'
OSM_NUM_TOKENS   = 64
LLM_HIDDEN_SIZE  = 4096

# Existing pipeline token counts (shallow, 2 frames, 2 CAVs)
LIDAR_TOKENS = (220 + 50) * 2 * 2   # (scene + object) × cavs × frames = 1080

# ─── Step 1: encoder ─────────────────────────────────────────────────────────

print("=" * 60)
print(f"Step 1: Loading {OSM_ENCODER_NAME} ...")

encoder = OSMViTEncoder(OSM_ENCODER_NAME, OSM_NUM_TOKENS)
encoder.eval()
print(f"  hidden_size : {encoder.hidden_size}")
print(f"  params      : {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M  (frozen)")

fake_osm = torch.randn(BATCH_SIZE, 3, 224, 224)
osm_tokens = encoder(fake_osm)

assert osm_tokens.shape == (BATCH_SIZE, OSM_NUM_TOKENS, encoder.hidden_size)
print(f"  output      : {tuple(osm_tokens.shape)}  ✓")

# ─── Step 2: projector ───────────────────────────────────────────────────────

print()
print("Step 2: Projector ...")

proj = build_osm_projector(encoder.hidden_size, LLM_HIDDEN_SIZE)
proj.eval()
osm_features = proj(osm_tokens)

assert osm_features.shape == (BATCH_SIZE, OSM_NUM_TOKENS, LLM_HIDDEN_SIZE)
print(f"  {encoder.hidden_size} → {LLM_HIDDEN_SIZE} : {tuple(osm_features.shape)}  ✓")

# ─── Step 3: concat with fake LiDAR tokens ───────────────────────────────────

print()
print("Step 3: Concat with LiDAR tokens ...")

fake_lidar = torch.randn(BATCH_SIZE, LIDAR_TOKENS, LLM_HIDDEN_SIZE)
combined   = torch.cat([fake_lidar, osm_features], dim=1)

assert combined.shape == (BATCH_SIZE, LIDAR_TOKENS + OSM_NUM_TOKENS, LLM_HIDDEN_SIZE)
print(f"  {LIDAR_TOKENS} LiDAR + {OSM_NUM_TOKENS} OSM = {LIDAR_TOKENS+OSM_NUM_TOKENS} tokens")
print(f"  combined : {tuple(combined.shape)}  ✓")

# ─── Step 4: context budget ──────────────────────────────────────────────────

print()
print("Step 4: Context budget ...")

total  = LIDAR_TOKENS + OSM_NUM_TOKENS + 500   # 500 = QA text estimate
margin = 2048 - total
print(f"  {LIDAR_TOKENS} + {OSM_NUM_TOKENS} + 500 (QA) = {total} / 2048  (margin {margin})")
assert margin >= 0, f"Context overflow by {-margin} tokens!"
print(f"  ✓")

# ─── Step 5: image transform (as in __getitem__) ─────────────────────────────

print()
print("Step 5: PIL → tensor transform ...")

from PIL import Image
from torchvision import transforms as T

fake_png   = Image.new('RGB', (256, 256), color=(180, 200, 220))
transform  = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
t = transform(fake_png)
assert t.shape == (3, 224, 224)
print(f"  256×256 PNG → {tuple(t.shape)}  ✓")

# ─── Done ────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("All steps passed — OSM pipeline is ready.")
