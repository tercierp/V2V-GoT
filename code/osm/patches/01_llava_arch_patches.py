"""
patches/01_llava_arch_patches.py
================================

Three surgical changes to /scratch/izar/.../V2V-GoT/LLaVA/llava/model/llava_arch.py
to add OSM token injection.

This file is documentation, not executable. Each change is shown as a
before/after snippet you can apply manually with `nano` or `vim`.

==============================================================================
PATCH A — In LlavaMetaModel.__init__ (around line 39)
==============================================================================

FIND this block (already exists):

            self.mm_scene_projector = build_scene_vision_projector(config)

ADD these lines RIGHT AFTER it (still inside the same `if` block):

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
                )

==============================================================================
PATCH B — Modify `generate_point_features` (line 699)
==============================================================================

CHANGE the function signature FROM:

    def generate_point_features(self, my_model_config, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask):

TO (added osm_features kwarg at the end, default None for backward compat):

    def generate_point_features(self, my_model_config, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask, osm_features=None):

Then INSIDE the function, replace EVERY `return ...` statement with a wrapped
version that prepends OSM tokens. There are THREE return points to patch:

  1. End of `if scene_level_only:` branch (around line 720):
     OLD:  return scene_level_features
     NEW:  return self._maybe_prepend_osm(scene_level_features, osm_features)

  2. End of `elif object_level_only:` branch (around line 731):
     OLD:  return object_level_features
     NEW:  return self._maybe_prepend_osm(object_level_features, osm_features)

  3. End of else (both scene and object) branch (around line 766):
     OLD:  return point_features
     NEW:  return self._maybe_prepend_osm(point_features, osm_features)

Then ADD this helper method anywhere in the class (recommended: just BEFORE
`generate_point_features`, around line 698):

    def _maybe_prepend_osm(self, point_features, osm_features):
        \"\"\"Prepend OSM tokens to the point feature sequence if enabled.

        Args:
            point_features: (B, N, 4096) existing scene+object tokens
            osm_features:   (B, 4, 200, 200) BEV tensor or None
        Returns:
            (B, num_osm + N, 4096) if OSM enabled, else unchanged.
        \"\"\"
        if osm_features is None or not getattr(self.get_model(), 'use_osm_tokens', False):
            return point_features
        # OSMEncoder lives on the inner model (LlavaMetaModel)
        osm_tokens = self.get_model().osm_encoder(osm_features)  # (B, 4, 4096)
        # Cast to the same dtype as the rest of the prompt embeddings
        osm_tokens = osm_tokens.to(point_features.dtype)
        return torch.cat([osm_tokens, point_features], dim=1)


==============================================================================
PATCH C — Plumb osm_features through prepare_inputs_labels_for_multimodal
==============================================================================

CHANGE the function signature (line 772-774) FROM:

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, my_model_config=None, scene_point_feature_map=None, regression_map=None, classification_map=None, detection_box_score=None, object_features=None, active_agent_mask=None
    ):

TO (added osm_features kwarg at the end):

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, my_model_config=None, scene_point_feature_map=None, regression_map=None, classification_map=None, detection_box_score=None, object_features=None, active_agent_mask=None,
        osm_features=None
    ):

Then INSIDE the function, find the call to `generate_point_features` (around
line 838) and add osm_features to the call:

OLD:
    point_features = self.generate_point_features(my_model_config, scene_point_feature_map, regression_map, classification_map,  detection_box_score, object_features, active_agent_mask)

NEW:
    point_features = self.generate_point_features(my_model_config, scene_point_feature_map, regression_map, classification_map,  detection_box_score, object_features, active_agent_mask, osm_features=osm_features)

==============================================================================
DONE — that's all the model-side changes for llava_arch.py.
"""
