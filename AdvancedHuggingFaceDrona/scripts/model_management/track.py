#!/usr/bin/env python3
"""
Model tracking for HuggingFaceDrona environment.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from utils import format_size


def setup_tracking():
    """Set up model tracking directory and file"""
    print("Setting up model tracking...")

    # Create tracking directory
    config_dir = Path.home() / ".drona" / "huggingFaceDrona"
    config_dir.mkdir(parents=True, exist_ok=True)

    models_file = config_dir / "models"

    # Initialize models file if it doesn't exist
    if not models_file.exists():
        with open(models_file, 'w') as f:
            json.dump({}, f, indent=2)
        print(f"Created tracking file: {models_file}")
    else:
        print(f"Using existing tracking file: {models_file}")

    return str(models_file)


def track_model(model_id, model_path, size_bytes=None):
    """Add a model to the tracking file"""
    config_dir = Path.home() / ".drona" / "huggingFaceDrona"
    config_dir.mkdir(parents=True, exist_ok=True)
    models_file = config_dir / "models"

    # Load existing models
    try:
        with open(models_file, 'r') as f:
            models = json.load(f)
    except:
        models = {}

    # Add new model entry
    models[model_id] = {
        "model_id": model_id,
        "path": str(model_path),
        "download_date": datetime.now().isoformat(),
        "size_bytes": size_bytes,
        "size_human": format_size(size_bytes) if size_bytes else "Unknown"
    }

    # Save updated models
    with open(models_file, 'w') as f:
        json.dump(models, f, indent=2)

    print(f"✓ Added {model_id} to tracking database")
    print(f"  Path: {model_path}")
    print(f"  Size: {format_size(size_bytes) if size_bytes else 'Unknown'}")


if __name__ == "__main__":
    # Can be used standalone for testing
    pass