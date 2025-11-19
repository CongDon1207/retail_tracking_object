import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings:
    # Paths
    MODEL_NAME = os.getenv("MODEL_NAME", "yolo11l.pt")
    VIDEO_PATH = os.getenv("VIDEO_PATH", "data/video3.mp4")
    OUT_JSONL = os.getenv("OUT_JSONL", "metadata/video.jsonl")
    
    # Tracker
    TRACKER_TYPE = os.getenv("TRACKER_TYPE", "botsort")  # botsort | bytetrack
    CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
    
    # Filter
    # Parse string "[0, 1]" -> list [0, 1]
    _class_filter_str = os.getenv("CLASS_FILTER", "[0]")
    try:
        import json
        CLASS_FILTER = json.loads(_class_filter_str)
    except:
        CLASS_FILTER = [0]  # Default fallback

    # Metadata
    STORE_ID = os.getenv("STORE_ID", "store_01")
    CAMERA_ID = os.getenv("CAMERA_ID", "cam_01")
    STREAM_ID = os.getenv("STREAM_ID", "stream_01")

settings = Settings()
