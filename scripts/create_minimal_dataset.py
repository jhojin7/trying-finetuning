#!/usr/bin/env python3
"""
Create a minimal test dataset in COCO format for pipeline testing.

This creates a simple synthetic dataset with colored rectangles as objects
to verify that the training pipeline works correctly.
"""

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import random

def create_synthetic_image(width=640, height=480, num_objects=3):
    """
    Create a synthetic image with colored rectangles.

    Returns:
        image: PIL Image
        annotations: List of bounding boxes and labels
    """
    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    annotations = []
    colors = ['red', 'blue', 'green', 'yellow', 'orange']

    for i in range(num_objects):
        # Random position and size
        x1 = random.randint(50, width - 150)
        y1 = random.randint(50, height - 150)
        w = random.randint(50, 100)
        h = random.randint(50, 100)
        x2 = min(x1 + w, width - 10)
        y2 = min(y1 + h, height - 10)

        # Draw rectangle
        color = random.choice(colors)
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)

        # Store annotation
        annotations.append({
            'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: x, y, width, height
            'category_id': colors.index(color),
            'area': (x2 - x1) * (y2 - y1)
        })

    return img, annotations


def create_coco_dataset(output_dir, split='train', num_images=20):
    """
    Create a COCO format dataset.

    Args:
        output_dir: Output directory
        split: Dataset split (train/valid/test)
        num_images: Number of images to generate
    """
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # COCO categories
    categories = [
        {'id': 0, 'name': 'red', 'supercategory': 'shape'},
        {'id': 1, 'name': 'blue', 'supercategory': 'shape'},
        {'id': 2, 'name': 'green', 'supercategory': 'shape'},
        {'id': 3, 'name': 'yellow', 'supercategory': 'shape'},
        {'id': 4, 'name': 'orange', 'supercategory': 'shape'},
    ]

    images = []
    annotations = []
    annotation_id = 1

    for img_id in range(1, num_images + 1):
        # Create image
        img, img_annotations = create_synthetic_image(num_objects=random.randint(2, 5))

        # Save image
        img_filename = f"{split}_{img_id:04d}.jpg"
        img_path = split_dir / img_filename
        img.save(img_path)

        # Add to images list
        images.append({
            'id': img_id,
            'file_name': img_filename,
            'width': 640,
            'height': 480,
            'license': 1,
            'date_captured': '2025-10-24'
        })

        # Add annotations
        for ann in img_annotations:
            annotations.append({
                'id': annotation_id,
                'image_id': img_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': 0,
                'segmentation': []
            })
            annotation_id += 1

    # Create COCO JSON
    coco_data = {
        'info': {
            'description': 'Minimal Test Dataset for Pipeline Verification',
            'version': '1.0',
            'year': 2025,
            'contributor': 'Claude Code',
            'date_created': '2025-10-24'
        },
        'licenses': [
            {'id': 1, 'name': 'Public Domain', 'url': ''}
        ],
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # Save annotations
    annotations_path = split_dir / '_annotations.coco.json'
    with open(annotations_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ Created {split} split: {num_images} images, {len(annotations)} annotations")
    return annotations_path


def main():
    """Create minimal test dataset."""
    print("🎨 Creating minimal test dataset for pipeline verification...")
    print()

    output_dir = Path('/home/user/trying-finetuning/dataset')

    # Create splits
    create_coco_dataset(output_dir, 'train', num_images=50)
    create_coco_dataset(output_dir, 'valid', num_images=15)
    create_coco_dataset(output_dir, 'test', num_images=15)

    print()
    print("=" * 60)
    print("✅ Dataset created successfully!")
    print(f"📁 Location: {output_dir.absolute()}")
    print()
    print("📊 Dataset Structure:")
    print("  - train: 50 images (colored rectangles)")
    print("  - valid: 15 images")
    print("  - test: 15 images")
    print("  - Categories: red, blue, green, yellow, orange")
    print()
    print("⚠️  NOTE: This is a SYNTHETIC dataset for testing the pipeline only!")
    print("   Replace with real data for actual training.")
    print("=" * 60)


if __name__ == '__main__':
    main()
