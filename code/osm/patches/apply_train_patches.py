#!/usr/bin/env python3
"""
apply_train_patches.py
======================

Apply OSM data-loader patches to train.py atomically.

Performs FOUR patches:
  A. Add OSM fields to ModelArguments dataclass
  B. Load OSM .npy per sample (right after data_dict['object_features'] = ...)
  C. Add 'osm_features' to the collate function feature list
  D. Add OSM keys to the my_model_config dict in main()

Usage:
    python apply_train_patches.py [--dry-run] [--file PATH]

Same safety guarantees as apply_llava_arch_patches.py:
  - Refuses to apply if any target text is missing or non-unique
  - Writes a .bak backup before any change
  - --dry-run flag verifies without writing
"""

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_PATH = "/scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA/llava/train/train.py"


# ============================================================================
# Patch A: Add OSM fields to ModelArguments (after dataset_source)
# ============================================================================
# We anchor on the unique line `dataset_source: Optional[str] = field(...)`
# inside ModelArguments (line ~78). DataArguments has a copy of some of these
# fields too — we use the FIRST occurrence (ModelArguments).
PATCH_A_FIND = """    dataset_source: Optional[str] = field(default="v2v4real") # or v2xreal


@dataclass
class DataArguments:"""

PATCH_A_REPLACE = """    dataset_source: Optional[str] = field(default="v2v4real") # or v2xreal
    # === OSM injection ===
    osm_features_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root dir containing pre-computed OSM .npy files. "
                          "If None, OSM tokens are disabled."},
    )
    use_osm_tokens: bool = field(
        default=False,
        metadata={"help": "Enable OSM token injection. Requires --osm_features_root."},
    )
    osm_num_tokens: int = field(
        default=4,
        metadata={"help": "Number of OSM tokens to inject (must be a perfect square)."},
    )
    osm_in_channels: int = field(
        default=4,
        metadata={"help": "Number of channels in the OSM BEV tensor."},
    )


@dataclass
class DataArguments:"""


# ============================================================================
# Patch B: Load OSM .npy per sample.
# Anchor on the unique 4-line block ending with `data_dict['object_features']`.
# ============================================================================
PATCH_B_FIND = """        object_features_all_frames = torch.stack(object_features_all_frames, dim=0)
        #print('object_features_all_frames.shape: ', object_features_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=256]
        data_dict['object_features'] = object_features_all_frames"""

PATCH_B_REPLACE = """        object_features_all_frames = torch.stack(object_features_all_frames, dim=0)
        #print('object_features_all_frames.shape: ', object_features_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=256]
        data_dict['object_features'] = object_features_all_frames

        # === OSM features ===
        # Load pre-computed OSM BEV tensor for this sample if a feature root
        # was configured. We key on (scenario_index, global_timestamp_index, asker_cav_id).
        osm_features_root = getattr(self.model_args, 'osm_features_root', None)
        if osm_features_root is not None:
            sample_dict = self.list_data_dict[i]
            scenario = sample_dict.get('scenario_index', -1)
            ts = sample_dict.get('global_timestamp_index', -1)
            asker = sample_dict.get('asker_cav_id', 'ego')
            osm_path = os.path.join(
                osm_features_root,
                '%d_%d_%s.npy' % (scenario, ts, asker)
            )
            if os.path.exists(osm_path):
                osm_tensor = np.load(osm_path).astype(np.float32)  # (4, 200, 200)
            else:
                # Missing → zero tensor; encoder still produces (dull) tokens.
                osm_tensor = np.zeros((4, 200, 200), dtype=np.float32)
            data_dict['osm_features'] = torch.from_numpy(osm_tensor)"""


# ============================================================================
# Patch C: Add 'osm_features' to the collate function feature list.
# ============================================================================
PATCH_C_FIND = """        for data_feature_name in ['scene_point_feature_map', 'regression_map', 'classification_map', 'detection_box_score', 'object_features', 'active_agent_mask', 'i', 'global_timestamp_index', 'local_timestamp_index', 'qa_sub_type']:"""

PATCH_C_REPLACE = """        for data_feature_name in ['scene_point_feature_map', 'regression_map', 'classification_map', 'detection_box_score', 'object_features', 'active_agent_mask', 'i', 'global_timestamp_index', 'local_timestamp_index', 'qa_sub_type', 'osm_features']:"""


# ============================================================================
# Patch D: Add OSM keys to my_model_config dict.
# ============================================================================
PATCH_D_FIND = """        'feature_source': model_args.feature_source,
        'dataset_source': model_args.dataset_source,
    }"""

PATCH_D_REPLACE = """        'feature_source': model_args.feature_source,
        'dataset_source': model_args.dataset_source,
        'use_osm_tokens': getattr(model_args, 'use_osm_tokens', False),
        'osm_num_tokens': getattr(model_args, 'osm_num_tokens', 4),
        'osm_in_channels': getattr(model_args, 'osm_in_channels', 4),
    }"""


# ============================================================================
# Driver
# ============================================================================
ALL_PATCHES = [
    ("Patch A: add OSM fields to ModelArguments", PATCH_A_FIND, PATCH_A_REPLACE),
    ("Patch B: load OSM .npy per sample", PATCH_B_FIND, PATCH_B_REPLACE),
    ("Patch C: add 'osm_features' to collate list", PATCH_C_FIND, PATCH_C_REPLACE),
    ("Patch D: add OSM keys to my_model_config", PATCH_D_FIND, PATCH_D_REPLACE),
]


def apply(text: str) -> tuple[str, list[str]]:
    log = []
    for name, find, repl in ALL_PATCHES:
        n = text.count(find)
        if n == 0:
            raise RuntimeError(
                f"FAILED — {name}: target text not found.\n"
                f"Expected to find:\n----\n{find[:300]}\n----"
            )
        if n > 1:
            raise RuntimeError(
                f"FAILED — {name}: target text found {n} times (must be unique).\n"
                f"Expected unique:\n----\n{find[:300]}\n----"
            )
        text = text.replace(find, repl)
        log.append(f"  ✓ {name}")
    return text, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_PATH)
    ap.add_argument("--dry-run", action="store_true")
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
        print(f"\nWARNING: backup already exists at {backup} — not overwriting.")
    else:
        shutil.copy2(p, backup)
        print(f"\nWrote backup to {backup}")

    p.write_text(patched)
    print(f"Wrote patched file to {p}  ({len(patched)} bytes, +{len(patched) - len(original)})")
    print("\nNext step:")
    print("  cd /scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA")
    print("  python -c 'import llava.train.train; print(\"OK\")'")


if __name__ == "__main__":
    main()
