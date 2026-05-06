#!/usr/bin/env python3
"""Inference setup using drona_add_mapping for placeholders."""

from drona_utils import drona_add_mapping, drona_add_additional_file, drona_add_note, drona_add_error


def setup_inference(operation, model_id="", dataset_id="", dataset_source="", input_text="",
                   max_samples="5", hf_token="", location="", model_source="hub",
                   gpu_type="auto", num_gpus="1", nodes="1", precision="auto",
                   accelerator_type="gpu", **kwargs):
    """Setup inference using drona mappings."""

    # Validate
    if not model_id:
        drona_add_error("Model ID required")
        return ""

    if dataset_source not in ["hub", "downloaded", "text", "upload"]:
        drona_add_error("Invalid dataset source")
        return ""

    if dataset_source in ["hub", "downloaded"] and not dataset_id:
        drona_add_error("Dataset ID required for dataset source")
        return ""

    if dataset_source == "text" and not input_text:
        drona_add_error("Input text required for text source")
        return ""

    # Add inference file
    drona_add_additional_file("scripts/inference/inference.py", "inference.py")

    # Map simple placeholders
    drona_add_mapping("MODEL_ID", model_id)

    # Map device based on accelerator type
    if accelerator_type == "cpu":
        drona_add_mapping("DEVICE_MAP", '"cpu"')
        drona_add_mapping("PRECISION", "torch.float32")  # CPU uses float32
    else:
        drona_add_mapping("DEVICE_MAP", '"auto"')  # GPU auto placement
        drona_add_mapping("PRECISION", get_precision_code(precision))

    # Only add token if it's actually set (not empty or placeholder)
    if hf_token and hf_token.strip() and not hf_token.startswith("$"):
        drona_add_mapping("HF_TOKEN_ARG", f', token="{hf_token}"')
    else:
        drona_add_mapping("HF_TOKEN_ARG", "")

    # Map conditional imports
    if dataset_source in ["hub", "downloaded"]:
        drona_add_mapping("DATASET_IMPORT", "\nfrom datasets import load_dataset")
    else:
        drona_add_mapping("DATASET_IMPORT", "")

    # Map data loading section
    data_section = generate_data_loading_section(dataset_id, dataset_source, input_text, max_samples, hf_token)
    drona_add_mapping("DATA_LOADING_SECTION", data_section)

    # Map generation params
    gen_params = """max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id"""
    drona_add_mapping("GENERATION_PARAMS", gen_params)

    # Set command
    drona_add_mapping("OPERATION_CMD", "python inference.py")
    drona_add_mapping("OPERATION_DESC", f"Inference: {model_id}")

    # Add notes
    drona_add_note(f"Model: {model_id} (source: {model_source})")
    if dataset_source in ["hub", "downloaded"]:
        drona_add_note(f"Dataset: {dataset_id} ({dataset_source}, max {max_samples} samples)")
    elif dataset_source == "text":
        preview = input_text[:50] + "..." if len(input_text) > 50 else input_text
        drona_add_note(f"Input text: {preview}")

    # Add accelerator info
    if accelerator_type == "cpu":
        drona_add_note("Accelerator: CPU Only")
    elif accelerator_type == "gpu":
        drona_add_note(f"Accelerator: NVIDIA GPU ({gpu_type}, {num_gpus} GPU(s))")
        drona_add_note(f"Precision: {precision}")
    elif accelerator_type == "ipu":
        drona_add_note("Accelerator: Graphcore IPU (not yet implemented)")

    return ""


def get_precision_code(precision):
    """Get PyTorch dtype code based on precision setting."""
    precision_map = {
        "auto": "torch.float16 if torch.cuda.is_available() else torch.float32",
        "float16": "torch.float16",
        "float32": "torch.float32",
        "bfloat16": "torch.bfloat16"
    }
    return precision_map.get(precision, precision_map["auto"])


def generate_data_loading_section(dataset_id, dataset_source, input_text, max_samples, hf_token):
    """Generate data loading section based on source."""

    max_samples_int = int(max_samples) if max_samples else 5

    # Custom text - simple
    if dataset_source == "text":
        escaped_text = input_text.replace('"', '\\"').replace('\n', '\\n') if input_text else ""
        return f"""
    # Input text
    samples = ["{escaped_text}"]
"""

    # Dataset from hub or downloaded
    elif dataset_source in ["hub", "downloaded"]:
        # Only add token if it's actually set (not empty or placeholder like $hf_token)
        token_arg = f', token="{hf_token}"' if hf_token and hf_token.strip() and not hf_token.startswith("$") else ""
        return f"""
    # Load data
    print("Loading dataset...")
    dataset = load_dataset("{dataset_id}"{token_arg})

    # Get first split
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]

    # Find text column
    text_column = None
    for col in data.features:
        if data.features[col].dtype == 'string':
            text_column = col
            break

    if text_column is None:
        text_column = list(data.features.keys())[0]

    # Get samples
    samples = []
    for i, item in enumerate(data):
        if i >= {max_samples_int}:
            break
        samples.append(item[text_column])
"""

    else:
        return "\n    samples = []\n"
