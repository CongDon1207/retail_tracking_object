2025-11-07: Change BoT-SORT config at track/config/botsort.yaml - Enable ReID with OSNet, tune thresholds for static camera crowd; update setup.txt to add torchreid (completed).
2025-11-07: Fix tracking stability - Lower YOLO track conf to 0.20 in track/yolo_tracker_base.py and relax match/appearance in botsort.yaml for 1s occlusion cases (completed).
2025-11-07: Fix ReID init error - Set `model: auto` in track/config/botsort.yaml to use native YOLO features (Ultralytics expects YOLO .pt for ReID), resolving AttributeError (completed).
2025-11-07: Tune occlusion handling - Increase `track_buffer` to 120, relax `match_thresh` to 0.7 and `appearance_thresh` to 0.55 for ID stability across ~1s occlusion (completed).
2025-11-07: Add DeepSORT tuning - Read parameters from ENV (DS_MAX_AGE, DS_N_INIT, DS_MAX_IOU_DISTANCE, DEEPSORT_EMBEDDER, DS_DET_CONF) and document usage in README (completed).
2025-11-07: Retune BoT-SORT for occlusion - Lower proximity_thresh to 0.35, set new_track_thresh to 0.45, set gmc_method to none, raise appearance_thresh to 0.6 (camera static) (completed).
2025-11-07: Add YOLO_TRACK_CONF env - Allow overriding track conf via ENV in track/yolo_tracker_base.py for quick tuning (completed).
2025-11-19: Refactor codebase - Extract `utils/path_utils.py`, `emit/visualizer.py`, `config/settings.py` and refactor `main.py` to improve modularity and remove hardcoding (completed).
2025-11-19: Move Visualizer - Relocate `emit/visualizer.py` to `utils/visualizer.py` for better semantic organization (completed).
