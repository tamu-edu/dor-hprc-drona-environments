#!/usr/bin/env python3
"""
Utility functions for HuggingFaceDrona model management.
Contains helper functions for file operations, formatting, etc.
"""

import os
import argparse
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi, model_info

def get_directory_size(path):
    """Calculate total size of directory"""
    total_size = 0
    try:
        for f in Path(path).rglob('*'):
            if f.is_file():
                total_size += f.stat().st_size
    except Exception as e:
        print(f"Warning: Could not calculate directory size: {e}")
    return total_size

def count_files(path):
    """Count total files in directory"""
    try:
        return sum(1 for f in Path(path).rglob('*') if f.is_file())
    except Exception as e:
        print(f"Warning: Could not count files: {e}")
        return 0

def format_size(size_bytes):
    """Format bytes to human readable string"""
    if size_bytes is None or size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def check_directory_exists(path):
    """Check if directory exists and create if needed"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False

def get_model_name_from_id(model_id):
    """Extract model name from HuggingFace model ID"""
    return model_id.split('/')[-1]

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download Hugging Face models")
    parser.add_argument("--model-id", required=True, help="Model ID on Hugging Face Hub")
    parser.add_argument("--download-path", default="model_cache", help="Path to download the model")
    parser.add_argument("--token", help="Hugging Face token for gated models")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if model exists")
    return parser.parse_args()

def get_model_size(model_id, token=None):
    """Get the estimated size of a model"""
    try:
        api = HfApi(token=token)
        info = model_info(model_id, token=token)

        total_size = 0
        if hasattr(info, 'siblings') and info.siblings:
            for sibling in info.siblings:
                if hasattr(sibling, 'size') and sibling.size:
                    total_size += sibling.size

        return total_size if total_size > 0 else None
    except Exception as e:
        print(f"Warning: Could not get model size: {e}")
        return None

def check_disk_space(download_path, estimated_size):
    """Check if there's enough disk space for the download"""
    try:
        free_space = shutil.disk_usage(download_path).free
        if estimated_size and estimated_size > free_space:
            print(f"Warning: Estimated model size ({format_size(estimated_size)}) exceeds available disk space ({format_size(free_space)})")
            return False
        return True
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return True

def download_model_from_hub(model_id, download_path="model_cache", token=None, force_download=False):
    """
    Download a model from Hugging Face Hub
    Returns: (success, model_path, model_size)
    """
    try:
        print(f"Starting download of model: {model_id}")

        # Create download directory
        download_dir = os.path.abspath(os.path.expanduser(download_path))
        os.makedirs(download_dir, exist_ok=True)
        print(f"Download directory: {download_dir}")

        # Model will be saved in download_path/model_name
        model_name = model_id.split('/')[-1]
        model_local_path = os.path.join(download_dir, model_name)

        # Get model size estimate
        print("Checking model information...")
        estimated_size = get_model_size(model_id, token)
        if estimated_size:
            print(f"Estimated model size: {format_size(estimated_size)}")

            # Check disk space
            if not check_disk_space(download_dir, estimated_size):
                print("Error: Insufficient disk space for download")
                return False, None, None

        # Check if model already exists
        if os.path.exists(model_local_path) and not force_download:
            print(f"Model already exists at: {model_local_path}")
            print("Use --force-download to re-download")
            # Calculate existing size
            actual_size = get_directory_size(model_local_path)
            return True, model_local_path, actual_size

        print("Starting download...")

        # Download model snapshot
        model_path = snapshot_download(
            repo_id=model_id,
            token=token if token else None,
            local_dir=model_local_path,
            force_download=force_download
        )

        print("✓ Download completed!")
        print(f"Model saved to: {model_path}")

        # Calculate actual size and file count
        actual_size = get_directory_size(model_path)
        file_count = count_files(model_path)

        print(f"Total files: {file_count}")
        print(f"Download size: {format_size(actual_size)}")

        return True, model_path, actual_size

    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False, None, None