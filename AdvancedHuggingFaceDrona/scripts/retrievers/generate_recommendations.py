import sys

import math

import torch

import pandas as pd

from pathlib import Path

from urllib.parse import urlparse



# --- Accelerate/Hub Imports (from model_utils.py) ---

from accelerate.commands.estimate import create_empty_model

from accelerate.utils import calculate_maximum_sizes

from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError



# --- Configuration ---

H100_RAM_GB = 80

DTYPE_MODIFIER = {"float32": 1, "float16": 2, "bfloat16": 2, "int8": 4, "int4": 8}

PRECISION_OPTIONS = ["float32", "float16", "int8", "int4"] # Use float16 as the representative for 16-bit



# --- Helper Functions (Adapted from model_utils.py) ---



def get_model(model_name: str, library_name: str = "transformers"):

    """

    Finds and grabs a model from the Hub, initializing it on the 'meta' device.

    Exits gracefully with an error message if something goes wrong.

    """

    try:

        # trust_remote_code is often necessary for complex models

        return create_empty_model(model_name, library_name=library_name, trust_remote_code=True)

    except (GatedRepoError, HfHubHTTPError):

        print(f"Error: Model `{model_name}` is a gated repository. Please ensure you have provided access via `huggingface-cli login` in your environment.", file=sys.stderr)

        sys.exit(1)

    except RepositoryNotFoundError:

        print(f"Error: Model `{model_name}` was not found on the Hugging Face Hub.", file=sys.stderr)

        sys.exit(1)

    except Exception as e:

        print(f"An unexpected error occurred while loading the model `{model_name}`: {e}", file=sys.stderr)

        sys.exit(1)



def calculate_memory_needs(model: torch.nn.Module):

    """Calculates memory usage for a model on the 'meta' device."""

    total_size_bytes, _ = calculate_maximum_sizes(model)

    data = []

    for dtype in PRECISION_OPTIONS:

        modifier = DTYPE_MODIFIER.get(dtype, 1)

        dtype_total_size_bytes = total_size_bytes / modifier

        

        # Calculate sizes in GB

        base_model_gb = dtype_total_size_bytes / (1024**3)

        inference_gb = base_model_gb * 1.2  # Standard 1.2x buffer for inference

        

        # Full training requires 4x model size for Adam optim states + gradients

        # Not applicable for int8/int4

        full_train_gb = base_model_gb * 4 if dtype in ["float32", "float16"] else 0

        

        data.append({

            "precision": dtype,

            "inference_gb": inference_gb,

            "full_train_gb": full_train_gb,

        })

    return pd.DataFrame(data).set_index('precision')



def get_status_html(num_gpus, task_name):

    """Returns a styled paragraph with an icon based on the number of GPUs."""

    if task_name == "Full Fine-Tune" and num_gpus == 0:

        return '<span style="color: #6c757d;">N/A</span>' # Gray for N/A



    if num_gpus == 1:

        icon, color = "✅", "#28a745" # Green

    elif num_gpus <= 4:

        icon, color = "⚠️", "#ffc107" # Yellow

    else:

        icon, color = "⛔️", "#dc3545" # Red

        

    label = f"{num_gpus} GPU" if num_gpus == 1 else f"{num_gpus} GPUs"

    return f'<span style="color: {color}; font-weight: 500;">{icon} {label}</span>'



def print_html_output(model_name: str, recommendations: pd.DataFrame):

    """Prints the final, styled HTML snippet to stdout."""

    print("""

<div class="recommendation-container">

<style>

    .recommendation-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; background-color: #fff; }

    .recommendation-container h4 { margin-top: 0; margin-bottom: 15px; font-size: 16px; color: #495057; }

    .recommendation-container table { width: 100%; border-collapse: collapse; font-size: 13px; }

    .recommendation-container th, .recommendation-container td { border: 1px solid #e9ecef; padding: 12px; text-align: center; }

    .recommendation-container th { background-color: #f8f9fa; font-weight: 600; white-space: nowrap; }

    .recommendation-container td:first-child { text-align: left; font-weight: 500; }

    .recommendation-container .task-label { font-weight: bold; color: #6c757d; font-size: 11px; display: block; margin-bottom: 4px; text-transform: uppercase; }

    .recommendation-container code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; background-color: #e9ecef; padding: 3px 6px; border-radius: 4px; }

</style>""")

    print(f"<h4>H100 (80GB) GPU Recommendations for <code>{model_name}</code></h4>")

    print("<table><thead><tr><th>Precision</th><th>Inference</th><th>Full Fine-Tune</th></tr></thead><tbody>")



    for precision, row in recommendations.iterrows():

        print(f"<tr><td>{precision}</td>")

        print(f"<td>{get_status_html(row['inference_gpus'], 'Inference')}</td>")

        print(f"<td>{get_status_html(row['full_train_gpus'], 'Full Fine-Tune')}</td></tr>")



    print("</tbody></table></div>")



# --- Main Execution ---

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Error: Model name must be provided as an argument.", file=sys.stderr)

        sys.exit(1)

        

    model_name = sys.argv[1]

    

    # 1. Create the empty model object

    model = get_model(model_name)

    

    # 2. Calculate memory requirements

    memory_needs_df = calculate_memory_needs(model)

    

    # 3. Calculate GPU counts for H100

    memory_needs_df['inference_gpus'] = memory_needs_df['inference_gb'].apply(lambda x: math.ceil(x / H100_RAM_GB))

    memory_needs_df['full_train_gpus'] = memory_needs_df['full_train_gb'].apply(lambda x: math.ceil(x / H100_RAM_GB) if x > 0 else 0)

    

    # 4. Generate and print the final HTML

    print_html_output(model_name, memory_needs_df)
