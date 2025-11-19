Current Status
- Enabled BoT-SORT ReID; switched ReID `model` to `auto` to use native YOLO features (Ultralytics does not accept OSNet .pth here).
- Tuned botsort.yaml for crowded retail: longer track_buffer, adjusted thresholds, disabled GMC.
- Lowered YOLO .track() conf to 0.20 to keep detections near occlusion edges.
- DeepSORT tracker now reads ENV for tuning: DS_MAX_AGE, DS_N_INIT, DS_MAX_IOU_DISTANCE, DEEPSORT_EMBEDDER, DS_DET_CONF.
 - BoT-SORT occlusion tuning updated: `proximity_thresh=0.35`, `new_track_thresh=0.45`, `gmc_method=none`, `appearance_thresh=0.6`.
 - YOLO track conf can be overridden by ENV `YOLO_TRACK_CONF`.

Next Steps
- Install dependency: `pip install -r setup.txt` (adds torchreid).
- Run: `python main.py` (or point to your crowded video) and visually verify ID stability; measure ID switches.
- If ID merges in heavy crowd, raise `appearance_thresh` to 0.7–0.75.
- If tracks still drop during long occlusion, `track_buffer` is set to 120; increase to 150 if FPS > 30.
- If IDs still switch after ~1s occlusion, consider lowering `match_thresh` to 0.65 and keeping `appearance_thresh` at 0.55, then verify JSONL metrics.
 - To evaluate DeepSORT: set `tracker_type = "deepsort"` in `main.py` and use ENV, e.g. `DS_MAX_AGE=120 DS_MAX_IOU_DISTANCE=0.75 DEEPSORT_EMBEDDER=torchreid`.

Key Paths
- track/config/botsort.yaml
- track/config/weights/osnet_x0_5_msmt17.pth
- metadata/video.jsonl

Latest Checks
- metadata/video.jsonl currently empty; no prior run to analyze.

Environment
- Python 3.9–3.11
- setup.txt updated (torchreid added)
