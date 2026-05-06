#!/usr/bin/env python3
"""
Retriever script to list downloaded datasets for dynamicSelect dropdown.
Returns datasets from the tracking file in value/label format for selection.
"""

import json
import os
import sys
from pathlib import Path

def format_size(size_bytes):
    """Format bytes to human readable string"""
    if size_bytes is None or size_bytes == 0:
        return "Unknown"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def check_dataset_exists(dataset_path):
    """Check if the dataset directory still exists"""
    return os.path.exists(dataset_path) and os.path.isdir(dataset_path)

def get_downloaded_datasets_select():
    """Get list of downloaded datasets for dynamicSelect"""
    try:
        # Path to tracking file
        config_dir = Path.home() / ".drona" / "huggingFaceDrona"
        datasets_file = config_dir / "datasets"

        if not datasets_file.exists():
            return json.dumps([])

        # Load datasets data
        with open(datasets_file, 'r') as f:
            datasets_data = json.load(f)

        if not datasets_data:
            return json.dumps([])

        # Build options list
        options = []

        # Sort datasets by download date (newest first)
        sorted_datasets = sorted(
            datasets_data.values(),
            key=lambda x: x.get('download_date', ''),
            reverse=True
        )

        for dataset in sorted_datasets:
            dataset_id = dataset.get('dataset_id', 'Unknown')
            dataset_path = dataset.get('path', '')
            size_human = dataset.get('size_human', 'Unknown')

            # Only include datasets that still exist
            if not check_dataset_exists(dataset_path):
                continue

            # Format display label
            display_name = dataset_id.split('/')[-1] if '/' in dataset_id else dataset_id
            org_name = dataset_id.split('/')[0] if '/' in dataset_id else ''

            if org_name and org_name != display_name:
                label = f"{display_name} ({org_name}) - {size_human}"
            else:
                label = f"{display_name} - {size_human}"

            options.append({
                "value": dataset_id,
                "label": label
            })

        return json.dumps(options, indent=2)

    except Exception as e:
        # Return empty array on error
        return json.dumps([])

if __name__ == "__main__":
    print(get_downloaded_datasets_select())
