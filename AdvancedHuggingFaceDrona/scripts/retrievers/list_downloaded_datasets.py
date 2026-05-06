#!/usr/bin/env python3
"""
Retriever script to display downloaded datasets from the tracking file.
Returns HTML formatted list of datasets that have been downloaded via the dataset management workflow.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def format_size(size_bytes):
    """Format bytes to human readable string"""
    if size_bytes is None or size_bytes == 0:
        return "Unknown"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def format_date(iso_date_str):
    """Format ISO date string to readable format"""
    try:
        dt = datetime.fromisoformat(iso_date_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return iso_date_str

def check_dataset_exists(dataset_path):
    """Check if the dataset directory still exists"""
    return os.path.exists(dataset_path) and os.path.isdir(dataset_path)

def get_downloaded_datasets():
    """Get list of downloaded datasets from tracking file"""
    try:
        # Path to tracking file
        config_dir = Path.home() / ".drona" / "huggingFaceDrona"
        datasets_file = config_dir / "datasets"

        if not datasets_file.exists():
            return generate_empty_state_html()

        # Load datasets data
        with open(datasets_file, 'r') as f:
            datasets_data = json.load(f)

        if not datasets_data:
            return generate_empty_state_html()

        # Generate HTML for datasets list
        html = generate_datasets_list_html(datasets_data)
        return html

    except Exception as e:
        return f"""
        <div style='background: #f8d7da; border: 1px solid #f5c6cb; padding: 12px; margin: 10px 0; border-radius: 4px;'>
            <div style='color: #721c24; font-size: 0.9em;'>
                <strong>Error loading downloaded datasets:</strong> {str(e)}
            </div>
        </div>
        """

def generate_empty_state_html():
    """Generate HTML for when no datasets are downloaded"""
    return """
    <details style='margin: 10px 0; border: 1px solid #dee2e6; border-radius: 4px;'>
        <summary style='padding: 8px 12px; cursor: pointer; background: #f8f9fa; font-size: 0.9em; color: #495057; user-select: none;'>
            📁 Downloaded Datasets (0)
        </summary>
        <div style='padding: 12px; color: #6c757d; font-size: 0.85em; text-align: center;'>
            No datasets downloaded yet. Use the dataset selector above to download your first dataset.
        </div>
    </details>
    """

def generate_datasets_list_html(datasets_data):
    """Generate HTML list of downloaded datasets"""
    html = """
    <details style='margin: 10px 0; border: 1px solid #dee2e6; border-radius: 4px;'>
        <summary style='padding: 8px 12px; cursor: pointer; background: #f8f9fa; font-size: 0.9em; color: #495057; user-select: none;'>
            📁 Downloaded Datasets ({count})
        </summary>
        <div style='padding: 12px;'>
    """.format(count=len(datasets_data))

    # Sort datasets by download date (newest first)
    sorted_datasets = sorted(
        datasets_data.values(),
        key=lambda x: x.get('downloaded_at', ''),
        reverse=True
    )

    for dataset in sorted_datasets:
        dataset_id = dataset.get('dataset_id', 'Unknown')
        dataset_path = dataset.get('path', '')
        downloaded_at = dataset.get('downloaded_at', '')
        size_bytes = dataset.get('size_bytes')
        size_human = format_size(size_bytes) if size_bytes else 'Unknown'

        # Check if dataset still exists
        exists = check_dataset_exists(dataset_path)
        status_icon = "✅" if exists else "❌"
        status_color = "#28a745" if exists else "#dc3545"
        status_text = "Available" if exists else "Missing"

        # Format dataset name for display
        display_name = dataset_id.split('/')[-1] if '/' in dataset_id else dataset_id
        org_name = dataset_id.split('/')[0] if '/' in dataset_id else ''

        html += f"""
            <div style='padding: 8px; margin: 4px 0; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center;'>
                <div style='flex: 1;'>
                    <div style='font-size: 0.9em; color: #495057; margin-bottom: 3px;'>
                        {status_icon} <strong>{display_name}</strong>
                        {f'<span style="color: #6c757d; font-size: 0.8em;">({org_name})</span>' if org_name else ''}
                    </div>
                    <div style='font-size: 0.75em; color: #6c757d;'>
                        <span style='color: {status_color};'>{status_text}</span> •
                        {size_human} •
                        {format_date(downloaded_at)}
                    </div>
                    <div style='font-size: 0.7em; color: #adb5bd; margin-top: 2px; font-family: monospace;'>
                        📂 {dataset_path}
                    </div>
                </div>
            </div>
        """

    html += """
        </div>
    </details>
    """

    return html

if __name__ == "__main__":
    # This retriever doesn't need any parameters
    print(get_downloaded_datasets())
