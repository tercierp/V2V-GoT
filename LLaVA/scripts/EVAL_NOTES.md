# V2V-GoT Unified Eval Notes

- Join key: `(qa_source, id)` from the GT JSON. Fallback: `(qa_type_id, id)` if `qa_source` is missing.
- Matching radius (F1): `--tau` selects the F1 threshold from the official evaluator's thresholds `[0.5, 1, 2, 4]`. Default is 2.0.
- Collision check: uses 3D IOU between a fixed-size CAV box (1.5 x 2 x 4) and GT boxes in the future ego coordinate system, per `check_has_collision` in the official evaluator.
- L2 for Q5/Q7: uses `l2_error_avg_03_all` when available (average of current-location and 3s error for single-waypoint prediction), otherwise `l2_error_avg_3s`.
