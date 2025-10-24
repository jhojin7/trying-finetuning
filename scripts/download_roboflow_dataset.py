#!/usr/bin/env python3
"""
Roboflow Dataset Download Script

This script downloads datasets from Roboflow Universe and prepares them for training.

Usage:
    # Download with API key from environment variable
    export ROBOFLOW_API_KEY="your_api_key_here"
    python scripts/download_roboflow_dataset.py

    # Or specify API key directly
    python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY

    # Download to custom location
    python scripts/download_roboflow_dataset.py --output-dir /path/to/dataset

    # Use different dataset
    python scripts/download_roboflow_dataset.py --workspace ata-92vi7 --project stationary-irjqd --version 1

Dataset Information:
    - Current dataset: Stationary Detection
    - Source: https://universe.roboflow.com/ata-92vi7/stationary-irjqd
    - Format: COCO JSON (compatible with Detectron2)
    - Workspace: ata-92vi7
    - Project: stationary-irjqd

Notes:
    - API key can be found at: https://app.roboflow.com/settings/api
    - Free tier allows limited downloads per month
    - Dataset will be downloaded in COCO format for Detectron2 compatibility
    - Existing dataset directory will be backed up if it exists
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

try:
    from roboflow import Roboflow
except ImportError:
    print("ERROR: roboflow package not found.")
    print("Install with: uv pip install --system roboflow")
    sys.exit(1)


def backup_existing_dataset(dataset_dir: Path) -> None:
    """
    Backup existing dataset directory if it exists.

    Args:
        dataset_dir: Path to the dataset directory
    """
    if dataset_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = dataset_dir.parent / f"{dataset_dir.name}_backup_{timestamp}"

        print(f"📦 Backing up existing dataset to: {backup_dir}")
        shutil.move(str(dataset_dir), str(backup_dir))
        print(f"✅ Backup complete: {backup_dir}")


def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: str,
    format: str = "coco"
) -> Path:
    """
    Download dataset from Roboflow.

    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
        output_dir: Output directory for dataset
        format: Dataset format (default: coco)

    Returns:
        Path to downloaded dataset directory
    """
    print(f"🔑 Authenticating with Roboflow...")
    rf = Roboflow(api_key=api_key)

    print(f"📊 Accessing project: {workspace}/{project}")
    project_obj = rf.workspace(workspace).project(project)

    print(f"📥 Downloading version {version} in {format.upper()} format...")
    dataset = project_obj.version(version).download(format, location=output_dir)

    return Path(dataset.location)


def verify_dataset_structure(dataset_dir: Path) -> bool:
    """
    Verify that the downloaded dataset has the expected structure.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        True if structure is valid, False otherwise
    """
    print(f"\n🔍 Verifying dataset structure in: {dataset_dir}")

    required_splits = ["train", "valid", "test"]
    all_valid = True

    for split in required_splits:
        split_dir = dataset_dir / split
        annotations_file = split_dir / "_annotations.coco.json"

        if not split_dir.exists():
            print(f"  ❌ Missing {split} directory")
            all_valid = False
            continue

        if not annotations_file.exists():
            print(f"  ❌ Missing annotations file in {split}")
            all_valid = False
            continue

        # Count images
        image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        print(f"  ✅ {split}: {len(image_files)} images, annotations present")

    return all_valid


def print_dataset_summary(dataset_dir: Path) -> None:
    """
    Print summary of dataset contents.

    Args:
        dataset_dir: Path to the dataset directory
    """
    import json

    print(f"\n📊 Dataset Summary")
    print(f"="*50)

    splits = ["train", "valid", "test"]
    for split in splits:
        annotations_file = dataset_dir / split / "_annotations.coco.json"

        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                data = json.load(f)

            num_images = len(data.get('images', []))
            num_annotations = len(data.get('annotations', []))
            categories = data.get('categories', [])

            print(f"\n{split.upper()}:")
            print(f"  Images: {num_images}")
            print(f"  Annotations: {num_annotations}")
            print(f"  Categories: {len(categories)}")

            if categories:
                print(f"  Class names:")
                for cat in categories:
                    print(f"    - {cat['name']} (id: {cat['id']})")


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from Roboflow Universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download with API key from environment
    export ROBOFLOW_API_KEY="your_api_key_here"
    python scripts/download_roboflow_dataset.py

    # Download with inline API key
    python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY

    # Download different version
    python scripts/download_roboflow_dataset.py --version 2

Getting Your API Key:
    1. Sign up at https://app.roboflow.com/
    2. Go to Settings > API
    3. Copy your API key
    4. Set it as environment variable or pass via --api-key

Current Dataset:
    https://universe.roboflow.com/ata-92vi7/stationary-irjqd
        """
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env variable)"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="ata-92vi7",
        help="Roboflow workspace name (default: ata-92vi7)"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="stationary-irjqd",
        help="Roboflow project name (default: stationary-irjqd)"
    )

    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset version number (default: 1)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Output directory for dataset (default: dataset)"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="coco",
        choices=["coco", "yolov5", "yolov8", "voc"],
        help="Dataset format (default: coco)"
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't backup existing dataset"
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        print("❌ ERROR: No API key provided!")
        print("\nPlease either:")
        print("  1. Set environment variable: export ROBOFLOW_API_KEY='your_key'")
        print("  2. Pass via argument: --api-key YOUR_KEY")
        print("\nGet your API key at: https://app.roboflow.com/settings/api")
        sys.exit(1)

    output_path = Path(args.output_dir)

    # Backup existing dataset
    if not args.no_backup:
        backup_existing_dataset(output_path)

    try:
        # Download dataset
        print(f"\n🚀 Starting download from Roboflow Universe")
        print(f"📍 Workspace: {args.workspace}")
        print(f"📦 Project: {args.project}")
        print(f"🔢 Version: {args.version}")
        print(f"💾 Format: {args.format.upper()}")
        print(f"📁 Output: {output_path.absolute()}")
        print()

        dataset_dir = download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=str(output_path.parent),
            format=args.format
        )

        # Verify structure
        if verify_dataset_structure(dataset_dir):
            print(f"\n✅ Dataset structure validated successfully!")
        else:
            print(f"\n⚠️  Dataset structure validation failed - please check manually")

        # Print summary
        if args.format == "coco":
            print_dataset_summary(dataset_dir)

        print(f"\n" + "="*50)
        print(f"✅ Download complete!")
        print(f"📁 Dataset location: {dataset_dir.absolute()}")
        print(f"\n🚀 Next steps:")
        print(f"   1. Review dataset: ls -la {dataset_dir}/train/")
        print(f"   2. Start training: jupyter notebook")
        print(f"   3. Open: notebooks/detectron2_finetuning.ipynb")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your API key is valid")
        print("  2. Verify project URL: https://universe.roboflow.com/{workspace}/{project}")
        print("  3. Ensure you have access to the project")
        print("  4. Check your internet connection")
        sys.exit(1)


if __name__ == "__main__":
    main()
