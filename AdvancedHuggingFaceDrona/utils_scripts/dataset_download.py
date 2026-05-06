#!/usr/bin/env python3

from drona_utils import drona_add_mapping, drona_add_additional_file, drona_add_note, drona_add_warning, drona_add_error

def setup_dataset_download(dataset_id="", cache_option="", hf_token="", location="", dataset_download_path="dataset_cache"):
    """Handle dataset download operation setup with modular scripts."""
    if not validate(dataset_id, cache_option, hf_token, location, dataset_download_path):
        return ""


    # Determine force download option
    force_download = (cache_option == "refresh")

    # Add the modular files using drona_add_additional_file
    drona_add_additional_file("scripts/dataset_management/download.py", "download.py")
    drona_add_additional_file("scripts/dataset_management/track.py", "track.py")
    drona_add_additional_file("scripts/dataset_management/utils.py", "utils.py")

    # Set operation mappings for the modular approach
    token_flag = f"--token {hf_token}" if hf_token else ""
    force_flag = "--force" if force_download else ""
    drona_add_mapping("OPERATION_CMD", f"python3 download.py --dataset-id '{dataset_id}' --download-path '{dataset_download_path}' {token_flag} {force_flag}".strip())
    drona_add_mapping("OPERATION_DESC", f"Dataset Management: {dataset_id}")

    # Add dataset-specific mappings for placeholders
    drona_add_mapping("DATASET_ID", dataset_id)
    drona_add_mapping("DOWNLOAD_PATH", dataset_download_path)
    drona_add_mapping("TOKEN_FLAG", token_flag)

    return ""


def validate(dataset_id, cache_option, hf_token, location, dataset_download_path):
    # Input validation and warnings
    if not dataset_id or dataset_id.strip() == "":
        drona_add_error("Dataset ID is required. Please select a dataset from HuggingFace Hub.")
        return False

    if not dataset_download_path or dataset_download_path.strip() == "":
        dataset_download_path = "dataset_cache"
        drona_add_warning(f"No download path specified. Datasets will be saved to default location: {location}/{dataset_download_path}")
    else:
        drona_add_note(f"Datasets will be saved to: {location}/{dataset_download_path}")

    if not location or location.strip() == "":
        drona_add_warning("Job location not set. Using current working directory.")

    # Add informational notes based on dataset selection
    drona_add_note(f"Setting up dataset management for: {dataset_id}")

    # Check if this looks like a gated dataset and warn about token
    if any(keyword in dataset_id.lower() for keyword in ['private', 'gated']) and not hf_token:
        drona_add_warning(f"Dataset '{dataset_id}' may be gated and require a HuggingFace token. If download fails, please provide your HF token.")

    # Provide cache behavior info
    if cache_option == "refresh":
        drona_add_note("Force download enabled - will re-download even if dataset exists locally.")
    elif cache_option == "use_cache":
        drona_add_note("Using cached version if available - will skip download if dataset already exists locally.")
    else:
        drona_add_note("Default caching behavior - will download if not present locally.")
    return True
