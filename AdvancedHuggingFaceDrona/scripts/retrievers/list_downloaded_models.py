#!/usr/bin/env python3
"""
Retriever script to display downloaded models from the tracking file.
Returns HTML formatted list of models that have been downloaded via the model management workflow.
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

def check_model_exists(model_path):
    """Check if the model directory still exists"""
    return os.path.exists(model_path) and os.path.isdir(model_path)

def get_downloaded_models():
    """Get list of downloaded models from tracking file"""
    try:
        # Path to tracking file
        config_dir = Path.home() / ".drona" / "huggingFaceDrona"
        models_file = config_dir / "models"
        
        if not models_file.exists():
            return generate_empty_state_html()
        
        # Load models data
        with open(models_file, 'r') as f:
            models_data = json.load(f)
        
        if not models_data:
            return generate_empty_state_html()
        
        # Generate HTML for models list
        html = generate_models_list_html(models_data)
        return html
        
    except Exception as e:
        return f"""
        <div style='background: #f8d7da; border: 1px solid #f5c6cb; padding: 12px; margin: 10px 0; border-radius: 4px;'>
            <div style='color: #721c24; font-size: 0.9em;'>
                <strong>Error loading downloaded models:</strong> {str(e)}
            </div>
        </div>
        """

def generate_empty_state_html():
    """Generate HTML for when no models are downloaded"""
    return """
    <details style='margin: 10px 0; border: 1px solid #dee2e6; border-radius: 4px;'>
        <summary style='padding: 8px 12px; cursor: pointer; background: #f8f9fa; font-size: 0.9em; color: #495057; user-select: none;'>
            📁 Downloaded Models (0)
        </summary>
        <div style='padding: 12px; color: #6c757d; font-size: 0.85em; text-align: center;'>
            No models downloaded yet. Use the model selector above to download your first model.
        </div>
    </details>
    """

def generate_models_list_html(models_data):
    """Generate HTML list of downloaded models"""
    html = """
    <details style='margin: 10px 0; border: 1px solid #dee2e6; border-radius: 4px;'>
        <summary style='padding: 8px 12px; cursor: pointer; background: #f8f9fa; font-size: 0.9em; color: #495057; user-select: none;'>
            📁 Downloaded Models ({count})
        </summary>
        <div style='padding: 12px;'>
    """.format(count=len(models_data))
    
    # Sort models by download date (newest first)
    sorted_models = sorted(
        models_data.values(), 
        key=lambda x: x.get('download_date', ''), 
        reverse=True
    )
    
    for model in sorted_models:
        model_id = model.get('model_id', 'Unknown')
        model_path = model.get('path', '')
        download_date = model.get('download_date', '')
        size_human = model.get('size_human', 'Unknown')
        
        # Check if model still exists
        exists = check_model_exists(model_path)
        status_icon = "✅" if exists else "❌"
        status_color = "#28a745" if exists else "#dc3545"
        status_text = "Available" if exists else "Missing"
        
        # Format model name for display
        display_name = model_id.split('/')[-1] if '/' in model_id else model_id
        org_name = model_id.split('/')[0] if '/' in model_id else ''
        
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
                        {format_date(download_date)}
                    </div>
                    <div style='font-size: 0.7em; color: #adb5bd; margin-top: 2px; font-family: monospace;'>
                        📂 {model_path}
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
    print(get_downloaded_models())