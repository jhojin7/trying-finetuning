#!/usr/bin/env python3
"""
Fix COCO annotation files to have valid category IDs starting from 1.

This script corrects the invalid category IDs that start from 0, which
violates COCO format specification and causes Detectron2 warnings.
"""

import json
import os
from pathlib import Path


def fix_coco_annotations(input_file: str, output_file: str = None) -> None:
    """
    Fix COCO annotations by remapping category IDs to start from 1.

    Args:
        input_file: Path to original COCO JSON file
        output_file: Path to save fixed JSON (defaults to overwriting input)
    """
    if output_file is None:
        output_file = input_file

    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create mapping from old ID to new ID
    old_to_new = {}
    for cat in data['categories']:
        old_id = cat['id']
        new_id = old_id + 1  # Shift all IDs by 1 to start from 1
        old_to_new[old_id] = new_id
        cat['id'] = new_id
        print(f"  Category '{cat['name']}': {old_id} -> {new_id}")

    # Update all annotation category_id references
    updated_count = 0
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        ann['category_id'] = old_to_new[old_cat_id]
        updated_count += 1

    # Save fixed annotations
    print(f"\nUpdated {updated_count} annotations")
    print(f"Saving to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Fixed! Category IDs now range from 1 to {len(data['categories'])}")


def main():
    # Get dataset root
    script_dir = Path(__file__).parent
    dataset_root = script_dir.parent / "dataset"

    # Fix all COCO annotation files
    annotation_files = [
        dataset_root / "train" / "_annotations.coco.json",
        dataset_root / "valid" / "_annotations.coco.json",
        dataset_root / "test" / "_annotations.coco.json",
    ]

    for anno_file in annotation_files:
        if anno_file.exists():
            print(f"\n{'='*60}")
            print(f"Processing: {anno_file.name} ({anno_file.parent.name})")
            print('='*60)

            # Create backup
            backup_file = anno_file.with_suffix('.json.backup')
            if not backup_file.exists():
                import shutil
                shutil.copy2(anno_file, backup_file)
                print(f"Backup created: {backup_file.name}")

            # Fix the file
            fix_coco_annotations(str(anno_file))
        else:
            print(f"⚠️  File not found: {anno_file}")

    print(f"\n{'='*60}")
    print("✅ All COCO annotation files have been fixed!")
    print("   Category IDs now start from 1 as required by COCO spec")
    print("   Original files backed up with .backup extension")
    print('='*60)


if __name__ == "__main__":
    main()
