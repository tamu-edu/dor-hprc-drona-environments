#!/usr/bin/env python3

import sys
import json
import os
from huggingface_hub import HfApi

def get_dataset_info(dataset_id, subset=None):
    """Get dataset information and preview from HuggingFace Hub."""
    
    if not dataset_id:
        return "<div style='background: #e9ecef; padding: 10px; border-radius: 4px; margin: 10px 0; color: #6c757d; font-size: 0.9em;'>Select a dataset to view information and data preview</div>"
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Get dataset info
        dataset_info = api.dataset_info(dataset_id)
        
        # Initialize display variables
        size_info = "Unknown"
        splits_info = "Unknown"
        columns_info = "Unknown"
        sample_data = ""
        
        # Get basic info from dataset_info object
        if hasattr(dataset_info, 'downloads') and dataset_info.downloads:
            downloads = dataset_info.downloads
            if downloads > 1000000:
                size_info = f"{downloads/1000000:.1f}M downloads"
            elif downloads > 1000:
                size_info = f"{downloads/1000:.1f}K downloads"
            else:
                size_info = f"{downloads} downloads"
        
        # Try to get info from first parquet file
        if hasattr(dataset_info, 'siblings') and dataset_info.siblings:
            parquet_files = [f for f in dataset_info.siblings if f.rfilename.endswith('.parquet')]
            if parquet_files:
                total_size = sum(f.size for f in parquet_files if f.size)
                if total_size > 0:
                    size_mb = total_size / (1024 * 1024)
                    if size_mb < 1024:
                        size_info = f"{size_mb:.1f}MB"
                    else:
                        size_gb = size_mb / 1024
                        size_info = f"{size_gb:.1f}GB"
        
        # Get splits and columns info via HF datasets server API
        try:
            import urllib.request
            import urllib.error
            
            # Try to get dataset info from HF datasets server
            config_name = subset if subset else "default"
            info_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
            
            req = urllib.request.Request(info_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if 'dataset_info' in data:
                        dataset_details = data['dataset_info']
                        
                        # Find the right config
                        config_data = None
                        if isinstance(dataset_details, dict):
                            if config_name in dataset_details:
                                config_data = dataset_details[config_name]
                            elif 'default' in dataset_details:
                                config_data = dataset_details['default']
                            else:
                                # Take the first available config
                                config_data = list(dataset_details.values())[0]
                        
                        if config_data:
                            # Get splits info
                            if 'splits' in config_data:
                                splits = config_data['splits']
                                split_names = []
                                for split_name, split_data in splits.items():
                                    if 'num_examples' in split_data:
                                        split_names.append(f"{split_name} ({split_data['num_examples']:,} rows)")
                                    else:
                                        split_names.append(split_name)
                                splits_info = ", ".join(split_names)
                            
                            # Get features/columns info
                            if 'features' in config_data:
                                features = config_data['features']
                                column_names = []
                                for feature_name, feature_data in features.items():
                                    if isinstance(feature_data, dict) and 'dtype' in feature_data:
                                        dtype = feature_data['dtype']
                                        if isinstance(dtype, str):
                                            column_names.append(f"{feature_name} ({dtype})")
                                        else:
                                            column_names.append(feature_name)
                                    else:
                                        column_names.append(feature_name)
                                columns_info = ", ".join(column_names)
                            
                            # Update size info if available
                            if 'dataset_size' in config_data:
                                size_bytes = config_data['dataset_size']
                                size_mb = size_bytes / (1024 * 1024)
                                if size_mb < 1024:
                                    size_info = f"{size_mb:.1f}MB"
                                else:
                                    size_gb = size_mb / 1024
                                    size_info = f"{size_gb:.1f}GB"
        except Exception as e:
            pass
        
        # Try to get sample data
        try:
            import urllib.request
            import urllib.error
            
            config_name = subset if subset else "default"
            # Try to get first split for sample data
            first_split = "train"
            if splits_info != "Unknown" and "train" not in splits_info.lower():
                # Extract first split name
                if "(" in splits_info:
                    first_split = splits_info.split("(")[0].strip().split(",")[0].strip()
            
            rows_url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_id}&config={config_name}&split={first_split}&offset=0&length=3"
            
            req = urllib.request.Request(rows_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if 'rows' in data and data['rows']:
                        rows = data['rows']
                        if rows:
                            sample_data = "<br><strong>Sample Data:</strong><br>"
                            sample_data += "<div style='font-family: monospace; font-size: 0.85em; max-height: 150px; overflow-y: auto; background: #fafcff; padding: 8px; border-radius: 4px; margin-top: 4px;'>"
                            
                            for i, row in enumerate(rows[:3]):
                                sample_data += f"<div style='margin: 4px 0; padding: 4px; background: white; border-radius: 2px;'>"
                                sample_data += f"<strong>Row {i+1}:</strong><br>"
                                row_data = row.get('row', {})
                                for key, value in row_data.items():
                                    # Truncate long values
                                    str_value = str(value)
                                    if len(str_value) > 100:
                                        str_value = str_value[:97] + "..."
                                    sample_data += f"&nbsp;&nbsp;{key}: {str_value}<br>"
                                sample_data += "</div>"
                            
                            sample_data += "</div>"
        except Exception as e:
            pass
        
        # Get tags for additional info
        tags = []
        if hasattr(dataset_info, 'tags') and dataset_info.tags:
            # Filter relevant tags
            relevant_tags = ['text', 'image', 'audio', 'tabular', 'multimodal', 'video', 'language']
            tags = [tag for tag in dataset_info.tags if any(rt in tag.lower() for rt in relevant_tags)]
        
        # Build HTML response (using grey color scheme)
        html = f"""
        <div style='background: #f8f9fa; padding: 12px; border: 1px solid #dee2e6; margin: 10px 0; border-radius: 4px;'>
            <h5 style='color: #495057; margin: 0 0 8px 0; font-size: 1.0em;'>{dataset_id}</h5>
            <div style='color: #6c757d; font-size: 0.9em;'>
                <strong>Size:</strong> {size_info} • <strong>Splits:</strong> {splits_info}<br>
                <strong>Columns:</strong> {columns_info}
        """
        
        if tags:
            html += f"<br><strong>Tags:</strong> {', '.join(tags)}"
        
        # Add license info if available
        license_info = ""
        if hasattr(dataset_info, 'card_data') and dataset_info.card_data:
            card_data = dataset_info.card_data
            if 'license' in card_data:
                license_info = card_data['license']
        
        if hasattr(dataset_info, 'gated') and dataset_info.gated:
            license_info = "Gated dataset"
        
        if license_info:
            html += f"<br><strong>License:</strong> {license_info}"
        
        # Add recommendations
        recommendations = []
        if "GB" in size_info:
            try:
                size_num = float(size_info.split("GB")[0])
                if size_num > 10:
                    recommendations.append("Large dataset - consider streaming")
                elif size_num > 1:
                    recommendations.append("Medium dataset - ensure sufficient disk space")
            except:
                pass
        
        if any(tag in str(tags).lower() for tag in ['image', 'audio', 'video']):
            recommendations.append("Media dataset - may require special preprocessing")
        
        if recommendations:
            html += f"<br><strong>Recommendations:</strong> {' • '.join(recommendations)}"
        
        # Add sample data if available
        if sample_data:
            html += sample_data
        
        html += """
            </div>
        </div>
        """
        
        return html
        
    except Exception as e:
        return f"<div style='background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 4px; margin: 10px 0; color: #721c24; font-size: 0.9em;'>Error fetching dataset info: {str(e)}</div>"

if __name__ == "__main__":
    # Get dataset_id from command line argument or environment variable
    dataset_id = ""
    subset = None
    
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1].strip('"')
    else:
        dataset_id = os.environ.get('DATASET_ID', '').strip('"')
    
    if len(sys.argv) > 2:
        subset = sys.argv[2].strip('"')
    else:
        subset = os.environ.get('DATASET_SUBSET', '').strip('"')
        if not subset:
            subset = None
    
    if dataset_id:
        print(get_dataset_info(dataset_id, subset))
    else:
        print("<div style='background: #e9ecef; padding: 10px; border-radius: 4px; margin: 10px 0; color: #6c757d; font-size: 0.9em;'>Select a dataset to view information and data preview</div>")
