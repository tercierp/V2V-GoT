"""
patches/03_llava_llama_patches.py
=================================

Two changes to /scratch/izar/.../V2V-GoT/LLaVA/llava/model/language_model/llava_llama.py
to (a) wire OSM config into the model config object and (b) accept
osm_features in forward().

==============================================================================
PATCH A — Wire OSM config (around line 116)
==============================================================================

FIND (existing line):

    config.mm_scene_projector_input_size = my_model_config['mm_scene_projector_input_size']

ADD NEARBY (inside the same config-setup block):

    config.use_osm_tokens = my_model_config.get('use_osm_tokens', False)
    config.osm_num_tokens = my_model_config.get('osm_num_tokens', 4)
    config.osm_in_channels = my_model_config.get('osm_in_channels', 4)


==============================================================================
PATCH B — Accept osm_features in forward()
==============================================================================

This file has a `forward` method that currently calls
`prepare_inputs_labels_for_multimodal` with the existing kwargs.

Find the line that calls prepare_inputs_labels_for_multimodal — typically in
the model's forward() method. It looks something like:

    def forward(
        self,
        input_ids: ...,
        ...
        scene_point_feature_map: Optional[torch.FloatTensor] = None,
        regression_map: Optional[torch.FloatTensor] = None,
        classification_map: Optional[torch.FloatTensor] = None,
        ...
    ):
        ...
        input_ids, position_ids, ... = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, past_key_values, labels,
            images, ...,
            scene_point_feature_map=scene_point_feature_map,
            ...,
        )

ADD `osm_features` at TWO places:

  1. As a kwarg in the forward method signature:
     osm_features: Optional[torch.FloatTensor] = None,

  2. In the call to prepare_inputs_labels_for_multimodal:
     osm_features=osm_features,

Use grep to find the forward method:

    grep -n "def forward" /scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA/llava/model/language_model/llava_llama.py

It usually appears once or twice in this file — the relevant one is the one
that calls prepare_inputs_labels_for_multimodal. Search for that call:

    grep -n "prepare_inputs_labels_for_multimodal" /scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA/llava/model/language_model/llava_llama.py

Both places need osm_features=osm_features added.
"""
