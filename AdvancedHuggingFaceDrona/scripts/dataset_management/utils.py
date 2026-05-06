#!/usr/bin/env python3
"""
Utility functions for dataset management operations.
"""

import os
from pathlib import Path


def format_size(size_bytes):
    """
    Format size in bytes to human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted size string (e.g., "1.5 GB")
    """
    if size_bytes is None:
        return "Unknown"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_dataset_size(dataset_path):
    """
    Calculate the total size of a dataset directory.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        int: Total size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"Warning: Could not calculate dataset size: {e}")
        return 0
    return total_size


def validate_dataset_id(dataset_id):
    """
    Validate dataset ID format.

    Args:
        dataset_id: HuggingFace dataset identifier

    Returns:
        bool: True if valid, False otherwise
    """
    if not dataset_id or dataset_id.strip() == "":
        return False

    # Basic validation - dataset IDs are typically username/dataset-name or just dataset-name
    # Allow alphanumeric, hyphens, underscores, and forward slashes
    import re
    pattern = r'^[a-zA-Z0-9_\-]+(/[a-zA-Z0-9_\-]+)?$'
    return bool(re.match(pattern, dataset_id))


def sanitize_path(dataset_id):
    """
    Convert dataset ID to a valid filesystem path component.

    Args:
        dataset_id: HuggingFace dataset identifier

    Returns:
        str: Sanitized path component
    """
    return dataset_id.replace("/", "_").replace(" ", "_")


def check_disk_space(path, required_bytes=None):
    """
    Check available disk space at a given path.

    Args:
        path: Path to check
        required_bytes: Optional required space in bytes

    Returns:
        dict: Dictionary with 'available', 'total', and 'sufficient' keys
    """
    try:
        stat = os.statvfs(path)
        available = stat.f_bavail * stat.f_frsize
        total = stat.f_blocks * stat.f_frsize

        result = {
            'available': available,
            'available_formatted': format_size(available),
            'total': total,
            'total_formatted': format_size(total),
        }

        if required_bytes:
            result['required'] = required_bytes
            result['required_formatted'] = format_size(required_bytes)
            result['sufficient'] = available >= required_bytes

        return result
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return {
            'available': 0,
            'available_formatted': 'Unknown',
            'total': 0,
            'total_formatted': 'Unknown'
        }


if __name__ == "__main__":
    # Test utilities
    print("Dataset Utils Test")
    print(f"Format 1.5 GB: {format_size(1500000000)}")
    print(f"Validate 'squad': {validate_dataset_id('squad')}")
    print(f"Validate 'huggingface/squad': {validate_dataset_id('huggingface/squad')}")
    print(f"Sanitize 'huggingface/squad': {sanitize_path('huggingface/squad')}")
    print(f"Disk space: {check_disk_space('.')}")
