#!/usr/bin/env python3
"""
Convert balanced COCO dataset to YOLO format.
"""

import json
import shutil
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
COCO_DATASET_ROOT = PROJECT_ROOT / 'dataset_balanced'
YOLO_DATASET_ROOT = PROJECT_ROOT / 'dataset_yolo_balanced'

def convert_coco_to_yolo(coco_json_path, output_dir, split_name):
    """
    Convert COCO format annotations to YOLO format.

    Args:
        coco_json_path: Path to COCO JSON annotation file
        output_dir: Root directory for YOLO dataset
        split_name: 'train', 'val', or 'test'
    """
    print(f"\nConverting {split_name} split...")

    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    images_dir = output_dir / 'images' / split_name
    labels_dir = output_dir / 'labels' / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build image id to filename mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Build category id mapping (COCO might start from 0 or 1, YOLO needs 0-indexed)
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    category_id_to_yolo_id = {cat['id']: idx for idx, cat in enumerate(categories)}
    category_names = [cat['name'] for cat in categories]

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Convert each image
    source_image_dir = Path(coco_json_path).parent
    converted_count = 0

    for image_id, image_info in image_id_to_info.items():
        image_filename = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        # Copy image
        source_image_path = source_image_dir / image_filename
        dest_image_path = images_dir / image_filename

        if source_image_path.exists():
            shutil.copy2(source_image_path, dest_image_path)
        else:
            print(f"⚠️  Image not found: {source_image_path}")
            continue

        # Convert annotations to YOLO format
        label_filename = Path(image_filename).stem + '.txt'
        label_path = labels_dir / label_filename

        yolo_annotations = []
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                # COCO bbox format: [x_min, y_min, width, height]
                x_min, y_min, bbox_width, bbox_height = ann['bbox']

                # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                x_center = (x_min + bbox_width / 2) / image_width
                y_center = (y_min + bbox_height / 2) / image_height
                norm_width = bbox_width / image_width
                norm_height = bbox_height / image_height

                # Get YOLO class id
                yolo_class_id = category_id_to_yolo_id[ann['category_id']]

                # YOLO format: class x_center y_center width height
                yolo_annotations.append(
                    f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                )

        # Write label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        converted_count += 1

    print(f"✅ Converted {converted_count} images for {split_name} split")
    return category_names

def main():
    print("="*80)
    print("CONVERTING BALANCED COCO DATASET TO YOLO FORMAT")
    print("="*80)

    # Check if COCO dataset exists
    if not COCO_DATASET_ROOT.exists():
        print(f"❌ Balanced COCO dataset not found at {COCO_DATASET_ROOT}")
        print("Please run balance_dataset.py first.")
        return

    # Convert all splits
    # Train
    train_json = COCO_DATASET_ROOT / 'train' / '_annotations.coco.json'
    if train_json.exists():
        class_names = convert_coco_to_yolo(train_json, YOLO_DATASET_ROOT, 'train')
    else:
        print(f"⚠️  Training annotations not found: {train_json}")
        return

    # Validation
    valid_json = COCO_DATASET_ROOT / 'valid' / '_annotations.coco.json'
    if valid_json.exists():
        convert_coco_to_yolo(valid_json, YOLO_DATASET_ROOT, 'val')
    else:
        print(f"⚠️  Validation annotations not found: {valid_json}")

    # Test
    test_json = COCO_DATASET_ROOT / 'test' / '_annotations.coco.json'
    if test_json.exists():
        convert_coco_to_yolo(test_json, YOLO_DATASET_ROOT, 'test')
    else:
        print(f"⚠️  Test annotations not found: {test_json}")

    # Create YOLO dataset YAML configuration
    dataset_yaml = {
        'path': str(YOLO_DATASET_ROOT.absolute()),  # Dataset root directory
        'train': 'images/train',  # Train images (relative to 'path')
        'val': 'images/val',      # Val images (relative to 'path')
        'test': 'images/test',    # Test images (relative to 'path')
        'nc': len(class_names),   # Number of classes
        'names': class_names      # Class names
    }

    # Save dataset configuration
    dataset_yaml_path = YOLO_DATASET_ROOT / 'dataset.yaml'
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*80}")
    print("✅ YOLO DATASET CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nYOLO dataset location: {YOLO_DATASET_ROOT}")
    print(f"Dataset YAML: {dataset_yaml_path}")
    print(f"\nDataset info:")
    print(f"  - Number of classes: {len(class_names)}")
    print(f"  - Classes: {class_names}")

    print("\n\nDataset YAML content:")
    with open(dataset_yaml_path, 'r') as f:
        print(f.read())

if __name__ == '__main__':
    main()
