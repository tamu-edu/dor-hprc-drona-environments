#!/usr/bin/env python3

import sys

import json

import requests

import os

from huggingface_hub import HfApi

def get_model_info(model_id):

    """Get model information from HuggingFace Hub."""

    if not model_id:

        return "<div style='background: #e9ecef; padding: 10px; border-radius: 4px; margin: 10px 0; color: #6c757d; font-size: 0.9em;'>Select a model to view size, type, and resource recommendations</div>"

    try:

        # Initialize HF API

        api = HfApi()

        # Get model info

        model_info = api.model_info(model_id)

        # Get model size from config if available

        size_info = "Unknown"

        model_type = "Unknown"

        vram_estimate = "Unknown"

        if hasattr(model_info, 'config') and model_info.config:

            config = model_info.config

            if 'model_type' in config:

                model_type = config['model_type'].title()

            # Estimate parameters from common config keys

            if 'n_parameters' in config:

                params = config['n_parameters']

                size_info = f"{params/1e9:.1f}B parameters"

            elif 'num_parameters' in config:

                params = config['num_parameters']

                size_info = f"{params/1e9:.1f}B parameters"

            elif 'hidden_size' in config and 'num_layers' in config:

                # Rough estimate for transformer models

                hidden_size = config['hidden_size']

                num_layers = config['num_layers']

                estimated_params = (hidden_size * hidden_size * num_layers * 12) / 1e9

                size_info = f"~{estimated_params:.1f}B parameters (estimated)"

        # Get file sizes if available

        if hasattr(model_info, 'siblings') and model_info.siblings:

            total_size = sum(file.size for file in model_info.siblings if file.size)

            if total_size > 0:

                size_gb = total_size / (1024**3)

                size_info = f"{size_gb:.1f}GB model size"

                # Estimate VRAM requirements (rough approximation)

                if size_gb < 2:

                    vram_estimate = "4-8GB VRAM"

                elif size_gb < 6:

                    vram_estimate = "8-16GB VRAM"

                elif size_gb < 15:

                    vram_estimate = "16-32GB VRAM"

                elif size_gb < 30:

                    vram_estimate = "32-48GB VRAM"

                else:

                    vram_estimate = "48GB+ VRAM"

        # Get tags for additional info

        tags = []

        if hasattr(model_info, 'tags') and model_info.tags:

            tags = [tag for tag in model_info.tags if tag in ['pytorch', 'tensorflow', 'flax', 'onnx', 'transformers']]

        # Build HTML response

        html = f"""

        <div style='background: #e8f5e8; border: 1px solid #c3e6c3; padding: 12px; margin: 10px 0; border-radius: 4px;'>

            <h5 style='color: #2d5a2d; margin: 0 0 8px 0; font-size: 1.0em;'>{model_id}</h5>

            <div style='color: #5a6c5a; font-size: 0.9em;'>

                <strong>Type:</strong> {model_type} • <strong>Size:</strong> {size_info}<br>

                <strong>Estimated VRAM:</strong> {vram_estimate}

        """

        if tags:

            html += f"<br><strong>Frameworks:</strong> {', '.join(tags)}"

        # Add recommendations

        recommendations = []

        if "32GB" in vram_estimate or "48GB" in vram_estimate:

            recommendations.append("Consider A100 or H100 GPUs")

        if "48GB+" in vram_estimate:

            recommendations.append("Enable quantization for smaller GPUs")

        if recommendations:

            html += f"<br><strong>Recommendations:</strong> {' • '.join(recommendations)}"

        html += """

            </div>

        </div>

        """

        return html

    except Exception as e:

        return f"<div style='background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 4px; margin: 10px 0; color: #721c24; font-size: 0.9em;'>Error fetching model info: {str(e)}</div>"

if __name__ == "__main__":
    # Get model_id from command line argument or environment variable
    model_id = ""
    if len(sys.argv) > 1:
        model_id = sys.argv[1].strip('"')
    else:
        model_id = os.environ.get('MODEL_ID', '').strip('"')
    
    if model_id:
        print(get_model_info(model_id))
    else:
        print("<div style='background: #e9ecef; padding: 10px; border-radius: 4px; margin: 10px 0; color: #6c757d; font-size: 0.9em;'>Select a model to view size, type, and resource recommendations</div>")


