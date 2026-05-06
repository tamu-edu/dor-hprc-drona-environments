#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from datasets import load_dataset
from track import track_dataset

def download_dataset(dataset_id, download_path="dataset_cache", revision=None, token=None, force=False):
    """Download a dataset from Hugging Face Hub"""
    print(f"Starting download of dataset: {dataset_id}")

    # Create download directory
    cache_dir = os.path.join(os.getcwd(), download_path)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using download directory: {cache_dir}")

    # Check if already downloaded
    dataset_path = os.path.join(cache_dir, dataset_id.replace("/", "_"))
    if os.path.exists(dataset_path) and not force:
        print(f"Dataset already exists at {dataset_path}")
        print("Use --force to re-download")
        return True

    try:
        # Download dataset
        print(f"Downloading dataset from HuggingFace Hub...")
        dataset = load_dataset(
            dataset_id,
            revision=revision if revision else None,
            cache_dir=cache_dir,
            token=token if token else None,
            trust_remote_code=True
        )

        # Save info about the dataset
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_id}")
        print(f"Splits: {list(dataset.keys())}")
        for split, data in dataset.items():
            print(f"  Split '{split}': {len(data)} samples")
        print(f"{'='*60}\n")

        # Calculate total size
        total_size = 0
        for split in dataset.values():
            if hasattr(split, 'data'):
                for batch in split.data.to_batches():
                    total_size += batch.nbytes

        print(f"Dataset download complete!")
        print(f"Location: {cache_dir}")

        # Track the downloaded dataset
        track_dataset(dataset_id, cache_dir, size_bytes=total_size if total_size > 0 else None)

        return True
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID on Hugging Face")
    parser.add_argument("--download-path", default="dataset_cache", help="Path to download the dataset")
    parser.add_argument("--revision", help="Dataset revision or tag")
    parser.add_argument("--token", help="Hugging Face token for gated datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download even if dataset exists")

    args = parser.parse_args()
    success = download_dataset(
        args.dataset_id,
        args.download_path,
        args.revision,
        args.token,
        args.force
    )
    sys.exit(0 if success else 1)
