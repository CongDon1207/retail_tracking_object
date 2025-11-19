Current Status
- Refactored codebase for modularity: extracted path logic to `utils/path_utils.py`, visualization to `utils/visualizer.py`, and configuration to `config/settings.py`.
- `main.py` is now cleaner and uses environment variables via `.env`.
- Enabled BoT-SORT ReID; switched ReID `model` to `auto` to use native YOLO features.
- Tuned botsort.yaml for crowded retail.

Next Steps
- Install dependency: `pip install -r setup.txt` (adds python-dotenv).
- Run: `python main.py` to verify tracking with new structure.
- Verify if `CVSource.py` integration is needed for better RTSP control.
- Continue tuning tracker parameters in `.env` or `botsort.yaml`.

Key Paths
- utils/path_utils.py
- utils/visualizer.py
- config/settings.py
- track/config/botsort.yaml
- metadata/video.jsonl

Latest Checks
- Refactor verification passed (imports and path resolution working).

Environment
- Python 3.9–3.11
- setup.txt updated (python-dotenv added)
