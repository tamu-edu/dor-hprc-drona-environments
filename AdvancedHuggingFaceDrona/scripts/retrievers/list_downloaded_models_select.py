#!/usr/bin/env python3
"""
Retriever script to list downloaded models for dynamicSelect dropdown.
Returns models from the tracking file in value/label format for selection.
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

def check_model_exists(model_path):
    """Check if the model directory still exists"""
    return os.path.exists(model_path) and os.path.isdir(model_path)

def get_downloaded_models_select():
    """Get list of downloaded models for dynamicSelect"""
    try:
        # Path to tracking file
        config_dir = Path.home() / ".drona" / "huggingFaceDrona"
        models_file = config_dir / "models"

        if not models_file.exists():
            return json.dumps([])

        # Load models data
        with open(models_file, 'r') as f:
            models_data = json.load(f)

        if not models_data:
            return json.dumps([])

        # Build options list
        options = []

        # Sort models by download date (newest first)
        sorted_models = sorted(
            models_data.values(),
            key=lambda x: x.get('download_date', ''),
            reverse=True
        )

        for model in sorted_models:
            model_id = model.get('model_id', 'Unknown')
            model_path = model.get('path', '')
            size_human = model.get('size_human', 'Unknown')

            # Only include models that still exist
            if not check_model_exists(model_path):
                continue

            # Format display label
            display_name = model_id.split('/')[-1] if '/' in model_id else model_id
            org_name = model_id.split('/')[0] if '/' in model_id else ''

            if org_name:
                label = f"{display_name} ({org_name}) - {size_human}"
            else:
                label = f"{display_name} - {size_human}"

            options.append({
                "value": model_id,
                "label": label
            })

        return json.dumps(options, indent=2)

    except Exception as e:
        # Return empty array on error
        return json.dumps([])

if __name__ == "__main__":
    print(get_downloaded_models_select())
