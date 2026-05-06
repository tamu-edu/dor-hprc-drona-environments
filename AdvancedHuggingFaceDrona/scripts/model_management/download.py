#!/usr/bin/env python3
"""
Simple model download script that uses modular components.
This script focuses only on the download orchestration.
"""

# Import our modular components
from utils import download_model_from_hub, get_args
from track import track_model

def main():
    """Main download orchestration"""
    # Get command line arguments
    args = get_args()

    print("=" * 60)
    print(f"Downloading model: {args.model_id}")
    print("=" * 60)

    # Download the model
    success, model_path, model_size = download_model_from_hub(
        model_id=args.model_id,
        download_path=args.download_path,
        token=args.token,
        force_download=args.force_download
    )

    if success:
        # Track the downloaded model
        track_model(args.model_id, model_path, model_size)
        print("\n✓ Model download and tracking completed successfully!")
        return 0
    else:
        print("\n✗ Model download failed!")
        return 1

if __name__ == "__main__":
    exit(main())