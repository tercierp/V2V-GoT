#!/usr/bin/env python3
"""
apply_llava_llama_patches.py
============================

Apply OSM patches to llava_llama.py atomically.

Performs THREE patches:
  A. Wire OSM config (use_osm_tokens, osm_in_channels, osm_num_tokens) into
     the model config object, right after the existing scene/object config wiring.
  B. Add osm_features kwarg to forward()
  C. Pass osm_features into the prepare_inputs_labels_for_multimodal calls
     (TWO call sites — generation and training paths)

Usage:
    python apply_llava_llama_patches.py [--dry-run] [--file PATH]
"""

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_PATH = "/scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA/llava/model/language_model/llava_llama.py"


# ============================================================================
# Patch A: Wire OSM config alongside existing scene/object/etc. config setup.
# Anchor on the existing `config.ego_only = my_model_config['ego_only']` line
# which is the last item in that block.
# ============================================================================
PATCH_A_FIND = """        config.mm_scene_projector_input_size = my_model_config['mm_scene_projector_input_size']    
        config.object_level_only = my_model_config['object_level_only']    
        config.scene_feature_mode = my_model_config['scene_feature_mode']    
        config.object_feature_mode = my_model_config['object_feature_mode']    
        config.ego_only = my_model_config['ego_only']    """

PATCH_A_REPLACE = """        config.mm_scene_projector_input_size = my_model_config['mm_scene_projector_input_size']    
        config.object_level_only = my_model_config['object_level_only']    
        config.scene_feature_mode = my_model_config['scene_feature_mode']    
        config.object_feature_mode = my_model_config['object_feature_mode']    
        config.ego_only = my_model_config['ego_only']    
        # === OSM injection ===
        config.use_osm_tokens = my_model_config.get('use_osm_tokens', False)
        config.osm_num_tokens = my_model_config.get('osm_num_tokens', 4)
        config.osm_in_channels = my_model_config.get('osm_in_channels', 4)"""


# ============================================================================
# Patches B + C: forward() signature and prepare_inputs_labels_for_multimodal calls.
#
# We can't write fixed-string find/replace blindly because we haven't seen the
# exact forward() signature lines. Instead, this script reads the file and
# does TWO things with structured edits:
#
#   (1) For EACH `prepare_inputs_labels_for_multimodal` call (there are TWO,
#       at lines 237 and 298): inject `osm_features=osm_features,` as a new
#       kwarg into the call.
#
#   (2) For the forward() at line 147: inject `osm_features=None,` into its
#       kwargs section.
#
# These structured edits use AST/textual heuristics rather than fixed strings.
# ============================================================================

import re


def patch_forward_signature(text: str) -> tuple[str, str]:
    """
    Find `def forward(` followed by the parameter list, and inject
    `osm_features: Optional[torch.FloatTensor] = None,` before the closing `):`.

    LLaVA-style forward() ends with one of these patterns:
        return_dict: Optional[bool] = None,
    ) -> ...:
    or just
    ):

    We use a regex that matches the closing `\\n    ) ->` or `\\n    ):` and
    inserts our kwarg right before it.
    """
    # Find the forward() definition
    forward_match = re.search(r"\n    def forward\(\s*\n", text)
    if not forward_match:
        raise RuntimeError("FAILED — Patch B: could not find 'def forward(' in llava_llama.py")
    start = forward_match.end()

    # Find the matching closing paren `\n    ) -> ...` or `\n    ):`
    end_match = re.search(r"\n    \) ->|\n    \):", text[start:])
    if not end_match:
        raise RuntimeError("FAILED — Patch B: could not find closing of forward() signature")
    end = start + end_match.start()

    sig_block = text[start:end]
    if "osm_features" in sig_block:
        return text, "Patch B: forward() already has osm_features (skipped)"

    # Inject kwarg right before the closing paren. The last existing kwarg may
    # or may not have a trailing comma — we add `        osm_features: Optional[torch.FloatTensor] = None,\n`
    # and let the closing `)` fall on the next line.
    # We also need to ensure indentation matches surrounding params (8 spaces).
    if not sig_block.endswith("\n"):
        injection = "\n        osm_features: Optional[torch.FloatTensor] = None,"
    else:
        injection = "        osm_features: Optional[torch.FloatTensor] = None,\n"
    new_text = text[:end] + injection + text[end:]
    return new_text, "Patch B: forward() signature accepts osm_features"


def patch_prepare_inputs_calls(text: str) -> tuple[str, str]:
    """
    Find calls to `self.prepare_inputs_labels_for_multimodal(` and inject
    `osm_features=osm_features` as a new kwarg in each call.

    Each call ends with `)`. The call may span multiple lines. We rely on
    matching balanced parentheses.

    IMPORTANT: the previous kwarg may or may not have a trailing comma.
    We always normalize by ensuring it ends with `,` before injecting.
    """
    pattern = re.compile(r"self\.prepare_inputs_labels_for_multimodal\(")
    matches = list(pattern.finditer(text))
    if not matches:
        raise RuntimeError("FAILED — Patch C: no calls to prepare_inputs_labels_for_multimodal found")

    # Walk in reverse so earlier offsets stay valid as we mutate
    n_patched = 0
    n_skipped = 0
    new_text = text
    for m in reversed(matches):
        call_start = m.end()  # position right after the opening `(`
        # Walk forward, tracking paren depth, until we find the matching `)`
        depth = 1
        i = call_start
        while i < len(new_text) and depth > 0:
            c = new_text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            raise RuntimeError("FAILED — Patch C: unbalanced parens in prepare_inputs_labels_for_multimodal call")

        call_block = new_text[call_start:i]
        if "osm_features" in call_block:
            n_skipped += 1
            continue

        # The call_block looks like:
        #   ...,\n                active_agent_mask\n            )
        # or
        #   ...,\n                active_agent_mask=active_agent_mask\n            )
        #
        # We need to insert `osm_features=osm_features,` BEFORE the closing `)`,
        # AND ensure the preceding line ends with a comma so the call stays valid.
        #
        # Strategy:
        #   1. Find the line that contains `)` (call_close_line_start..i)
        #   2. Walk back over that line's whitespace to find end-of-prev-line
        #   3. If prev-line non-whitespace tail does NOT end with a comma,
        #      insert one before the newline
        #   4. Then insert our new line at the indent of the closing `)`

        # Step 1: indent of the line containing `)`
        line_start = new_text.rfind("\n", 0, i) + 1
        close_indent = ""
        for ch in new_text[line_start:i]:
            if ch == " " or ch == "\t":
                close_indent += ch
            else:
                break

        # Step 2: end-of-prev-line offset (the character right before the
        # newline that ends the line *before* the line containing `)`)
        prev_line_end = line_start - 1  # position of '\n' that ends prev line
        if prev_line_end < call_start:
            # `)` is on the same line as `(` — that means this is a no-arg or
            # one-line call; we just inject before `)` directly.
            injection = "osm_features=osm_features"
            new_text = new_text[:i] + injection + new_text[i:]
            n_patched += 1
            continue

        # Walk back from prev_line_end to find the last non-whitespace char
        j = prev_line_end - 1
        while j > 0 and new_text[j] in (" ", "\t"):
            j -= 1
        last_char_of_prev_line = new_text[j]

        # Build the patch text:
        #   - if needed, add a comma after the last existing arg
        #   - then a newline + indented `osm_features=osm_features,`
        # We assemble by editing two locations in one go.
        prev_arg_kwarg_indent = close_indent + "    "  # default: 4 spaces deeper
        # Try to match the indent of the line we just walked back over so it
        # matches the existing args' indent.
        prev_line_start = new_text.rfind("\n", 0, prev_line_end) + 1
        prev_line_text = new_text[prev_line_start:prev_line_end]
        prev_line_indent = ""
        for ch in prev_line_text:
            if ch == " " or ch == "\t":
                prev_line_indent += ch
            else:
                break
        if prev_line_indent:
            prev_arg_kwarg_indent = prev_line_indent

        # Build replacement
        if last_char_of_prev_line == ",":
            # Comma already present — just inject the new arg line
            injection = f"{prev_arg_kwarg_indent}osm_features=osm_features,\n"
            new_text = new_text[:line_start] + injection + new_text[line_start:]
        else:
            # Need to add a comma after last arg, then inject new line
            # We do this by replacing: text[j+1..line_start] with: ",\n<injection>"
            # Original: text[j+1..line_start] is "<whitespace>\n"
            # New:      ",\n<indent>osm_features=osm_features,\n"
            # Preserve any trailing whitespace on prev line by re-appending it
            trailing_ws = new_text[j+1:prev_line_end]  # empty or whitespace
            replacement = (
                "," + trailing_ws + "\n"
                + prev_arg_kwarg_indent + "osm_features=osm_features,\n"
            )
            new_text = new_text[:j+1] + replacement + new_text[line_start:]
        n_patched += 1

    if n_patched == 0 and n_skipped > 0:
        msg = f"Patch C: all {n_skipped} call sites already had osm_features (skipped)"
    else:
        msg = f"Patch C: patched {n_patched} prepare_inputs_labels_for_multimodal calls (skipped {n_skipped})"
    return new_text, msg


# ============================================================================
# Driver
# ============================================================================

def apply(text: str) -> tuple[str, list[str]]:
    log = []

    # Patch A: fixed-string replacement
    n = text.count(PATCH_A_FIND)
    if n == 0:
        raise RuntimeError(
            f"FAILED — Patch A: target text not found.\n"
            f"Expected to find:\n----\n{PATCH_A_FIND[:300]}\n----"
        )
    if n > 1:
        raise RuntimeError(f"FAILED — Patch A: found {n} times (must be unique).")
    text = text.replace(PATCH_A_FIND, PATCH_A_REPLACE)
    log.append("  ✓ Patch A: wire OSM config (use_osm_tokens, etc.)")

    # Patch B: structured forward() signature edit
    text, msg = patch_forward_signature(text)
    log.append(f"  ✓ {msg}")

    # Patch C: structured prepare_inputs call edits
    text, msg = patch_prepare_inputs_calls(text)
    log.append(f"  ✓ {msg}")

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
    print(f"Wrote patched file to {p}  ({len(patched)} bytes, "
          f"+{len(patched) - len(original)})")
    print("\nNext step:")
    print("  cd /scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA")
    print("  python -c 'from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM; print(\"OK\")'")


if __name__ == "__main__":
    main()
