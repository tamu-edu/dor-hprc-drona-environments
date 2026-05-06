#!/usr/bin/env python3

from drona_utils import drona_add_mapping, drona_add_additional_file, drona_add_note, drona_add_warning, drona_add_error

def setup_model_download(model_id="", cache_option="", hf_token="", location="", model_download_path="model_cache"):
    """Handle model download operation setup with modular scripts."""
    if not validate(model_id, cache_option, hf_token, location, model_download_path):
        return ""
    
        
    # Determine force download option
    force_download = (cache_option == "refresh")
    
    # Add the modular files using drona_add_additional_file
    drona_add_additional_file("scripts/model_management/download.py", "download.py")
    drona_add_additional_file("scripts/model_management/track.py", "track.py")
    drona_add_additional_file("scripts/model_management/utils.py", "utils.py")
    
    # Set operation mappings for the modular approach
    drona_add_mapping("OPERATION_CMD", f"python3 download.py --model-id '{model_id}' --download-path '{model_download_path}'{' --token ' + hf_token if hf_token else ''}")
    drona_add_mapping("OPERATION_DESC", f"Model Management: {model_id}")

    # Add model-specific mappings for placeholders
    drona_add_mapping("MODEL_ID", model_id)
    drona_add_mapping("DOWNLOAD_PATH", model_download_path)
    drona_add_mapping("TOKEN_FLAG", f"--token {hf_token}" if hf_token else "")

    return ""


def validate(model_id, cache_option, hf_token, location, model_download_path):
    # Input validation and warnings
    if not model_id or model_id.strip() == "":
        drona_add_error("Model ID is required. Please select a model from HuggingFace Hub.")
        return False
    
    if not model_download_path or model_download_path.strip() == "":
        model_download_path = "model_cache"
        drona_add_warning(f"No download path specified. Models will be saved to default location: {location}/{model_download_path}")
    else:
        drona_add_note(f"Models will be saved to: {location}/{model_download_path}")
    
    if not location or location.strip() == "":
        drona_add_warning("Job location not set. Using current working directory.")
        
    # Add informational notes based on model selection
    drona_add_note(f"Downloading model: {model_id}")
    
    # Check if this looks like a gated model and warn about token
    if any(keyword in model_id.lower() for keyword in ['llama', 'claude', 'gpt', 'gemma']) and not hf_token:
        drona_add_warning(f"Model '{model_id}' may be gated and require a HuggingFace token. If download fails, please provide your HF token.")
    
    # Provide cache behavior info
    if cache_option == "refresh":
        drona_add_note("Force download enabled - will re-download even if model exists locally.")
    elif cache_option == "use_cache":
        drona_add_note("Using cached version if available - will skip download if model already exists locally.")
    else:
        drona_add_note("Default caching behavior - will download if not present locally.")
    return True

