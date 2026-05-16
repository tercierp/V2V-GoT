"""
patches/02_train_patches.py
===========================

Four small changes to /scratch/izar/.../V2V-GoT/LLaVA/llava/train/train.py
to load OSM features and plumb them through.

==============================================================================
PATCH A — Add new fields to ModelArguments / DataArguments (around line 74)
==============================================================================

FIND (existing):

    scene_feature_mode: Optional[str] = field(default="shallow")

ADD THESE FIELDS NEARBY (in the same dataclass, doesn't matter where):

    osm_features_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory containing pre-computed OSM .npy files. "
                          "If None, OSM tokens are disabled."}
    )
    use_osm_tokens: bool = field(
        default=False,
        metadata={"help": "Enable OSM token injection. Requires --osm_features_root."}
    )
    osm_num_tokens: int = field(
        default=4,
        metadata={"help": "Number of OSM tokens to inject (must be a perfect square)."}
    )
    osm_in_channels: int = field(
        default=4,
        metadata={"help": "Number of channels in the OSM BEV tensor."}
    )


==============================================================================
PATCH B — Make osm_features_root reachable from the dataset class (search
for the dataset class __init__; it usually receives data_args)
==============================================================================

In the dataset class __init__ (search for `class LazySupervisedDataset` or
similar V2V dataset), AFTER existing args are stored, add:

    self.osm_features_root = getattr(data_args, 'osm_features_root', None)


==============================================================================
PATCH C — Load OSM .npy per sample (around line 1389)
==============================================================================

FIND (existing line):

    data_dict['object_features'] = object_features_all_frames

ADD RIGHT AFTER:

    # === OSM features ===
    if self.osm_features_root is not None:
        sample_dict = self.list_data_dict[i]
        scenario = sample_dict.get('scenario_index', -1)
        ts = sample_dict.get('global_timestamp_index', -1)
        asker = sample_dict.get('asker_cav_id', 'ego')
        osm_path = os.path.join(
            self.osm_features_root,
            f"{scenario}_{ts}_{asker}.npy"
        )
        if os.path.exists(osm_path):
            osm_tensor = np.load(osm_path).astype(np.float32)  # (4, 200, 200)
        else:
            # Missing → zero tensor of the right shape so the model still
            # gets a tensor it can encode (encoder will produce dull tokens)
            osm_tensor = np.zeros((4, 200, 200), dtype=np.float32)
        data_dict['osm_features'] = torch.from_numpy(osm_tensor)


==============================================================================
PATCH D — Add 'osm_features' to the collate function feature list (line 1759)
==============================================================================

FIND (existing line):

    for data_feature_name in ['scene_point_feature_map', 'regression_map', 'classification_map', 'detection_box_score', 'object_features', 'active_agent_mask', 'i', 'global_timestamp_index', 'local_timestamp_index', 'qa_sub_type']:

ADD 'osm_features' to the list (anywhere is fine; recommended at end before 'qa_sub_type'):

    for data_feature_name in ['scene_point_feature_map', 'regression_map', 'classification_map', 'detection_box_score', 'object_features', 'active_agent_mask', 'i', 'global_timestamp_index', 'local_timestamp_index', 'qa_sub_type', 'osm_features']:


==============================================================================
PATCH E — Wire OSM config into the model config dict (around line 1916)
==============================================================================

FIND (existing line):

    'scene_feature_mode': model_args.scene_feature_mode,

ADD NEARBY:

    'use_osm_tokens': getattr(model_args, 'use_osm_tokens', False),
    'osm_num_tokens': getattr(model_args, 'osm_num_tokens', 4),
    'osm_in_channels': getattr(model_args, 'osm_in_channels', 4),


==============================================================================
PATCH F — Pass osm_features into model.forward
==============================================================================

This is where the ‘osm_features’ tensor in the batch dict reaches the model.
Search for where forward is called or where the batch is unpacked. In LLaVA
this is usually in `class LlavaTrainer` or in a custom forward hook. Look for
calls that pass `scene_point_feature_map`, `regression_map`, etc — you must
add `osm_features=batch.get('osm_features', None)` to those same calls.

The simplest place is the model's `forward` method in
`llava/model/language_model/llava_llama.py` — see patch file 03.
"""
