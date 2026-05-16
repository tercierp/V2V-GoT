#!/usr/bin/env python3
"""
apply_llava_arch_patches.py
===========================

Apply OSM token injection patches to llava_arch.py atomically.

Usage:
    python apply_llava_arch_patches.py [--dry-run] [--file PATH]

Default file path is the standard Izar location. Pass --dry-run to see what
would change without writing.

The script performs ALL THREE patches (A, B, C) in one shot. If any patch
fails to find its target, NO file changes are made — you get a clear error
telling you which patch failed and why.

A backup of the original file is written to {path}.bak before any change.
"""

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_PATH = "/scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA/llava/model/llava_arch.py"


# ============================================================================
# Patch A: In LlavaMetaModel.__init__, add OSM encoder build right after
# the mm_scene_projector instantiation.
# ============================================================================
PATCH_A_FIND = """            self.mm_scene_projector = build_scene_vision_projector(config)"""

PATCH_A_REPLACE = """            self.mm_scene_projector = build_scene_vision_projector(config)

            # === OSM injection ===
            # Build OSM encoder if config flag is set. Encoder converts the
            # pre-computed BEV tensor (4 channels, 200x200) into 4 LLM tokens
            # of dim 4096 (= config.hidden_size).
            self.use_osm_tokens = getattr(config, 'use_osm_tokens', False)
            if self.use_osm_tokens:
                from .osm_encoder import OSMEncoder
                self.osm_encoder = OSMEncoder(
                    in_channels=getattr(config, 'osm_in_channels', 4),
                    hidden_size=config.hidden_size,
                    num_tokens=getattr(config, 'osm_num_tokens', 4),
                )"""


# ============================================================================
# Patch B: generate_point_features signature + helper method + return rewrites.
#
# We match on the exact existing function signature and insert the helper
# method right before it, plus rewrite the signature to add osm_features.
# Then we replace the three return statements.
# ============================================================================
PATCH_B_SIG_FIND = """    def generate_point_features(self, my_model_config, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask):"""

PATCH_B_SIG_REPLACE = """    def _maybe_prepend_osm(self, point_features, osm_features):
        \"\"\"Prepend OSM tokens to the point feature sequence if enabled.

        Args:
            point_features: (B, N, 4096) existing scene+object tokens
            osm_features:   (B, 4, 200, 200) BEV tensor or None
        Returns:
            (B, num_osm + N, 4096) if OSM enabled, else unchanged.
        \"\"\"
        if osm_features is None or not getattr(self.get_model(), 'use_osm_tokens', False):
            return point_features
        osm_tokens = self.get_model().osm_encoder(osm_features)  # (B, 4, 4096)
        osm_tokens = osm_tokens.to(point_features.dtype)
        return torch.cat([osm_tokens, point_features], dim=1)

    def generate_point_features(self, my_model_config, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask, osm_features=None):"""


# Three return-statement replacements. Each must appear EXACTLY ONCE in the
# original file inside generate_point_features, but we keep the surrounding
# whitespace unique so the find string is anchored.
PATCH_B_RET1_FIND = """          scene_level_features = scene_level_features.reshape([batch_size, num_input_frames * num_cavs * num_tokens, feature_size])
          #print('scene_level_features.shape: ', scene_level_features.shape)
          #assert False
          return scene_level_features"""

PATCH_B_RET1_REPLACE = """          scene_level_features = scene_level_features.reshape([batch_size, num_input_frames * num_cavs * num_tokens, feature_size])
          #print('scene_level_features.shape: ', scene_level_features.shape)
          #assert False
          return self._maybe_prepend_osm(scene_level_features, osm_features)"""


PATCH_B_RET2_FIND = """          object_level_features = object_level_features.reshape([batch_size, num_input_frames * num_cavs * max_num_boxes_per_cav, feature_size])
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, 2*2*50, 4096]
          #assert False
          return object_level_features"""

PATCH_B_RET2_REPLACE = """          object_level_features = object_level_features.reshape([batch_size, num_input_frames * num_cavs * max_num_boxes_per_cav, feature_size])
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, 2*2*50, 4096]
          #assert False
          return self._maybe_prepend_osm(object_level_features, osm_features)"""


PATCH_B_RET3_FIND = """          # v2xreal [32, 1456 * num_input_frames, 4096]

          return point_features"""

PATCH_B_RET3_REPLACE = """          # v2xreal [32, 1456 * num_input_frames, 4096]

          return self._maybe_prepend_osm(point_features, osm_features)"""


# ============================================================================
# Patch C: prepare_inputs_labels_for_multimodal signature + inner call.
# ============================================================================
PATCH_C_SIG_FIND = """    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, my_model_config=None, scene_point_feature_map=None, regression_map=None, classification_map=None, detection_box_score=None, object_features=None, active_agent_mask=None
    ):"""

PATCH_C_SIG_REPLACE = """    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, my_model_config=None, scene_point_feature_map=None, regression_map=None, classification_map=None, detection_box_score=None, object_features=None, active_agent_mask=None,
        osm_features=None
    ):"""


PATCH_C_CALL_FIND = """                point_features = self.generate_point_features(my_model_config, scene_point_feature_map, regression_map, classification_map,  detection_box_score, object_features, active_agent_mask)"""

PATCH_C_CALL_REPLACE = """                point_features = self.generate_point_features(my_model_config, scene_point_feature_map, regression_map, classification_map,  detection_box_score, object_features, active_agent_mask, osm_features=osm_features)"""


# ============================================================================
# Driver
# ============================================================================

ALL_PATCHES = [
    ("Patch A: instantiate OSMEncoder in __init__", PATCH_A_FIND, PATCH_A_REPLACE),
    ("Patch B-sig: add osm_features kwarg + helper", PATCH_B_SIG_FIND, PATCH_B_SIG_REPLACE),
    ("Patch B-ret1: prepend osm in scene_only branch", PATCH_B_RET1_FIND, PATCH_B_RET1_REPLACE),
    ("Patch B-ret2: prepend osm in object_only branch", PATCH_B_RET2_FIND, PATCH_B_RET2_REPLACE),
    ("Patch B-ret3: prepend osm in combined branch", PATCH_B_RET3_FIND, PATCH_B_RET3_REPLACE),
    ("Patch C-sig: add osm_features to prepare_inputs", PATCH_C_SIG_FIND, PATCH_C_SIG_REPLACE),
    ("Patch C-call: pass osm_features into generate_point_features", PATCH_C_CALL_FIND, PATCH_C_CALL_REPLACE),
]


def apply(text: str, dry_run: bool = False) -> tuple[str, list[str]]:
    """Apply all patches. Raises RuntimeError on the first failure."""
    log = []
    for name, find, repl in ALL_PATCHES:
        n_occur = text.count(find)
        if n_occur == 0:
            raise RuntimeError(f"FAILED — {name}: target text not found.\n"
                               f"Expected to find:\n----\n{find[:200]}\n----")
        if n_occur > 1:
            raise RuntimeError(f"FAILED — {name}: target text found {n_occur} times "
                               "(must be unique).\n"
                               f"Expected unique:\n----\n{find[:200]}\n----")
        text = text.replace(find, repl)
        log.append(f"  ✓ {name}")
    return text, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_PATH,
                    help="Path to llava_arch.py")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not write changes, just verify all patches would apply")
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        sys.exit(f"ERROR: file does not exist: {p}")

    original = p.read_text()
    print(f"Loaded {p} ({len(original)} bytes)")

    try:
        patched, log = apply(original)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    print("\nAll patches verified. Summary:")
    for line in log:
        print(line)

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    backup = p.with_suffix(p.suffix + ".bak")
    if backup.exists():
        print(f"\nWARNING: backup already exists at {backup}")
        print("(Not overwriting; the existing .bak is your pre-patch original.)")
    else:
        shutil.copy2(p, backup)
        print(f"\nWrote backup to {backup}")

    p.write_text(patched)
    print(f"Wrote patched file to {p}  ({len(patched)} bytes, +{len(patched) - len(original)})")
    print("\nNext step: verify the file still imports.")
    print("  cd /scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA")
    print("  python -c 'from llava.model.llava_arch import LlavaMetaModel; print(\"OK\")'")


if __name__ == "__main__":
    main()
