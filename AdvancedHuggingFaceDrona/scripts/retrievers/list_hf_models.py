#!/usr/bin/env python3
"""
Script to list Hugging Face models for the Drona composer
This script can be called from the command line and returns a JSON list of models
"""

import sys
import json
import argparse
import requests

def list_hf_models(model_type="text-generation", search_query=""):
    """
    Fetch a list of popular Hugging Face models based on the selected task type
    and optional search query.
    
    Args:
        model_type: The type of model to search for (e.g., text-generation, image-classification)
        search_query: Optional user-entered search string to filter models
        
    Returns:
        List of model options for the dropdown
    """
    # Map model_type to Hugging Face API task parameter
    # Base URL for Hugging Face API
    base_url = "https://huggingface.co/api/models"
    
    # Set up query parameters
    params = {
        "sort": "downloads",
        "direction": -1,
        "limit": 50  # Fetch top 50 most downloaded models for this task
    }
    
    if search_query:
        params["search"] = search_query
    
    try:
        # Make request to Hugging Face API
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for non-200 responses
        
        models = response.json()
        
        # Format models for dropdown
        options = []
        for model in models:
            model_id = model.get("id", "")
            # Create a descriptive label with model ID and downloads count if available
            downloads = model.get("downloads", 0)
            downloads_str = f" ({downloads:,} downloads)" if downloads else ""
            
            options.append({
                "value": model_id,
                "label": f"{model_id}{downloads_str}"
            })
        
        return options
    
    except Exception as e:
        raise e
        


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='List Hugging Face models')
    parser.add_argument('--model_type', type=str, default='text-generation',
                        help='Type of model to search for')
    parser.add_argument('--search', type=str, default='',
                        help='Search query to filter models')
    
    args = parser.parse_args()
    
    # Get model list
    models = list_hf_models(args.model_type, args.search)
    
    # Print JSON output (will be captured by the calling process)
    print(json.dumps(models))

if __name__ == '__main__':
    main()
