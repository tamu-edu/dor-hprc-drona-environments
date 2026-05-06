#!/usr/bin/env python3
"""
Dataset tracking utilities for managing downloaded datasets.
Stores dataset metadata in ~/.drona/huggingFaceDrona/datasets
"""

import json
import os
from pathlib import Path
from datetime import datetime


def setup_tracking():
    """Initialize the tracking directory and file if they don't exist."""
    config_dir = Path.home() / ".drona" / "huggingFaceDrona"
    config_dir.mkdir(parents=True, exist_ok=True)

    datasets_file = config_dir / "datasets"
    if not datasets_file.exists():
        datasets_file.write_text("{}")

    return datasets_file


def track_dataset(dataset_id, dataset_path, size_bytes=None):
    """
    Add a dataset to the tracking file.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., 'squad', 'imdb')
        dataset_path: Local path where the dataset is stored
        size_bytes: Optional size of the dataset in bytes
    """
    datasets_file = setup_tracking()

    # Load existing datasets
    try:
        datasets = json.loads(datasets_file.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        datasets = {}

    # Add or update dataset entry
    datasets[dataset_id] = {
        "dataset_id": dataset_id,
        "path": str(dataset_path),
        "downloaded_at": datetime.now().isoformat(),
        "size_bytes": size_bytes
    }

    # Save updated datasets
    datasets_file.write_text(json.dumps(datasets, indent=2))
    print(f"✓ Dataset tracked: {dataset_id}")


def get_tracked_datasets():
    """
    Get all tracked datasets.

    Returns:
        dict: Dictionary of tracked datasets with dataset_id as key
    """
    datasets_file = setup_tracking()

    try:
        return json.loads(datasets_file.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def remove_dataset_tracking(dataset_id):
    """
    Remove a dataset from tracking.

    Args:
        dataset_id: HuggingFace dataset identifier to remove
    """
    datasets_file = setup_tracking()

    try:
        datasets = json.loads(datasets_file.read_text())
        if dataset_id in datasets:
            del datasets[dataset_id]
            datasets_file.write_text(json.dumps(datasets, indent=2))
            print(f"✓ Removed dataset from tracking: {dataset_id}")
            return True
    except (json.JSONDecodeError, FileNotFoundError):
        pass

    return False


if __name__ == "__main__":
    # Test the tracking functionality
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            datasets = get_tracked_datasets()
            if datasets:
                print("Tracked datasets:")
                for dataset_id, info in datasets.items():
                    print(f"  - {dataset_id}: {info['path']}")
            else:
                print("No tracked datasets")
        elif sys.argv[1] == "track" and len(sys.argv) >= 4:
            track_dataset(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == "remove" and len(sys.argv) >= 3:
            remove_dataset_tracking(sys.argv[2])
        else:
            print("Usage:")
            print("  python track.py list")
            print("  python track.py track <dataset_id> <path>")
            print("  python track.py remove <dataset_id>")
    else:
        print("Dataset tracking utility")
        print("Run with 'list', 'track', or 'remove' command")
