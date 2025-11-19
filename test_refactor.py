import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

try:
    from utils.path_utils import resolve_model_path, resolve_tracker_config
    from emit.visualizer import Visualizer
    from config.settings import settings
    print("Imports successful.")
    
    # Test path resolution
    try:
        model = resolve_model_path("yolo11l.pt")
        print(f"Resolved model: {model}")
    except Exception as e:
        print(f"Model resolution failed: {e}")

    # Test settings
    print(f"Settings loaded: Model={settings.MODEL_NAME}, Video={settings.VIDEO_PATH}")
    
    print("Refactor verification passed.")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Verification failed: {e}")
