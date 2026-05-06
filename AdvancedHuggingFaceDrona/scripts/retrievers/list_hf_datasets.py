#!/usr/bin/env python3
"""
Script to list Hugging Face datasets for the Drona composer
This script can be called from the command line and returns a JSON list of datasets
"""

import sys
import json
import argparse
import requests

def list_hf_datasets(dataset_type="", search_query=""):
    """
    Fetch a list of popular Hugging Face datasets based on the selected task type
    and optional search query.
    
    Args:
        dataset_type: The type of dataset to search for (e.g., text, image, audio)
        search_query: Optional user-entered search string to filter datasets
        
    Returns:
        List of dataset options for the dropdown
    """
    # Base URL for Hugging Face API
    base_url = "https://huggingface.co/api/datasets"
    
    # Set up query parameters
    params = {
        "sort": "downloads",
        "direction": -1,
        "limit": 50  # Fetch top 50 most downloaded datasets
    }
    
    if dataset_type:
        params["filter"] = dataset_type
        
    if search_query:
        params["search"] = search_query
    
    try:
        # Make request to Hugging Face API
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for non-200 responses
        
        datasets = response.json()
        
        # Format datasets for dropdown
        options = []
        for dataset in datasets:
            dataset_id = dataset.get("id", "")
            
            # Create a descriptive label with dataset ID and downloads count if available
            downloads = dataset.get("downloads", 0)
            downloads_str = f" ({downloads:,} downloads)" if downloads else ""
            
            # Include the tag if available
            tags = dataset.get("tags", [])
            tag_str = f" - {', '.join(tags[:3])}" if tags else ""
            
            options.append({
                "value": dataset_id,
                "label": f"{dataset_id}{downloads_str}{tag_str}"
            })
        
        return options
    
    except Exception as e:
        # Return error as an option
        return [{"value": "error", "label": f"Error loading datasets: {str(e)}"}]


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='List Hugging Face datasets')
    parser.add_argument('--dataset_type', type=str, default='',
                        help='Type of dataset to search for')
    parser.add_argument('--search', type=str, default='',
                        help='Search query to filter datasets')
    
    args = parser.parse_args()
    
    # Get dataset list
    datasets = list_hf_datasets(args.dataset_type, args.search)
    
    # Print JSON output (will be captured by the calling process)
    print(json.dumps(datasets))


if __name__ == '__main__':
    main()
