#!/usr/bin/env python3
"""
Create a balanced COCO dataset by resampling minority classes.
Target: minimum 200 annotations per class.
"""

import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / 'dataset'
BALANCED_ROOT = PROJECT_ROOT / 'dataset_balanced'

def balance_split(split_name, target_min=200):
    """
    Create a balanced version of a dataset split.

    Args:
        split_name: 'train', 'valid', or 'test'
        target_min: Minimum number of annotations per class
    """
    print(f"\n{'='*80}")
    print(f"Balancing {split_name.upper()} split (target: {target_min} annotations/class)")
    print(f"{'='*80}\n")

    # Load original annotations
    src_json = DATASET_ROOT / split_name / '_annotations.coco.json'
    src_images_dir = DATASET_ROOT / split_name

    with open(src_json, 'r') as f:
        data = json.load(f)

    # Create output directories
    dst_images_dir = BALANCED_ROOT / split_name
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    # Build mappings
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    # Count current annotations per category
    category_counts = Counter([ann['category_id'] for ann in data['annotations']])

    # Group images by categories they contain
    images_by_category = defaultdict(list)
    image_id_to_info = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        if img_id not in [img['image_id'] for img in images_by_category[cat_id]]:
            images_by_category[cat_id].append({
                'image_id': img_id,
                'image_info': image_id_to_info[img_id]
            })

    # Determine resampling strategy
    resampling_plan = {}
    for cat_id, count in category_counts.items():
        cat_name = cat_id_to_name[cat_id]

        if count < target_min:
            # Calculate how many times to duplicate images
            multiplier = int(target_min / count) + 1
            resampling_plan[cat_id] = {
                'name': cat_name,
                'current': count,
                'multiplier': multiplier,
                'target': count * multiplier
            }
        else:
            resampling_plan[cat_id] = {
                'name': cat_name,
                'current': count,
                'multiplier': 1,
                'target': count
            }

    # Print resampling plan
    print("Resampling plan:")
    print(f"{'Class':<30} {'Current':<12} {'Multiplier':<12} {'Target':<12}")
    print("-"*70)
    for cat_id in sorted(resampling_plan.keys(), key=lambda x: resampling_plan[x]['current']):
        plan = resampling_plan[cat_id]
        mult_str = f"{plan['multiplier']}x" if plan['multiplier'] > 1 else "1x (no change)"
        print(f"{plan['name']:<30} {plan['current']:<12} {mult_str:<12} {plan['target']:<12}")

    # Build new dataset
    new_images = []
    new_annotations = []
    next_image_id = 1
    next_ann_id = 1

    # Track which images we've already added
    added_original_images = set()

    # First, add all original images (multiplier=1 for all classes)
    for img_info in data['images']:
        img_id = img_info['id']

        # Copy image file
        src_img_path = src_images_dir / img_info['file_name']
        dst_img_path = dst_images_dir / img_info['file_name']

        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"⚠️  Image not found: {src_img_path}")
            continue

        # Add to new dataset with new ID
        new_img_info = img_info.copy()
        new_img_info['id'] = next_image_id
        new_images.append(new_img_info)

        # Map old ID to new ID
        old_to_new_img_id = {img_id: next_image_id}

        # Add annotations for this image
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                new_ann = ann.copy()
                new_ann['id'] = next_ann_id
                new_ann['image_id'] = next_image_id
                new_annotations.append(new_ann)
                next_ann_id += 1

        added_original_images.add(img_id)
        next_image_id += 1

    # Now, add duplicates for minority classes
    for cat_id, plan in resampling_plan.items():
        if plan['multiplier'] <= 1:
            continue  # No duplication needed

        cat_name = plan['name']
        print(f"\nDuplicating images for class: {cat_name} ({plan['multiplier']}x)")

        # Get images containing this category
        candidate_images = images_by_category[cat_id]

        # Duplicate (multiplier - 1) times (since we already added originals)
        for dup_idx in range(plan['multiplier'] - 1):
            for img_entry in candidate_images:
                old_img_id = img_entry['image_id']
                img_info = img_entry['image_info']

                # Create new filename for duplicate
                filename_parts = img_info['file_name'].rsplit('.', 1)
                new_filename = f"{filename_parts[0]}_dup{dup_idx+1}.{filename_parts[1]}"

                # Copy image file with new name
                src_img_path = src_images_dir / img_info['file_name']
                dst_img_path = dst_images_dir / new_filename

                if src_img_path.exists():
                    shutil.copy2(src_img_path, dst_img_path)
                else:
                    continue

                # Add to new dataset
                new_img_info = img_info.copy()
                new_img_info['id'] = next_image_id
                new_img_info['file_name'] = new_filename
                new_images.append(new_img_info)

                # Add annotations for this image
                for ann in data['annotations']:
                    if ann['image_id'] == old_img_id:
                        new_ann = ann.copy()
                        new_ann['id'] = next_ann_id
                        new_ann['image_id'] = next_image_id
                        new_annotations.append(new_ann)
                        next_ann_id += 1

                next_image_id += 1

    # Create new COCO JSON
    new_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories'],
        'info': data.get('info', {}),
        'licenses': data.get('licenses', [])
    }

    # Save JSON
    dst_json = BALANCED_ROOT / split_name / '_annotations.coco.json'
    with open(dst_json, 'w') as f:
        json.dump(new_data, f, indent=2)

    # Print statistics
    new_category_counts = Counter([ann['category_id'] for ann in new_annotations])

    print(f"\n✅ Balanced {split_name} split created!")
    print(f"   Original: {len(data['images'])} images, {len(data['annotations'])} annotations")
    print(f"   Balanced: {len(new_images)} images, {len(new_annotations)} annotations")
    print(f"\nNew class distribution:")
    print(f"{'Class':<30} {'Before':<12} {'After':<12} {'Change':<12}")
    print("-"*70)
    for cat_id in sorted(new_category_counts.keys(), key=lambda x: category_counts[x]):
        cat_name = cat_id_to_name[cat_id]
        before = category_counts[cat_id]
        after = new_category_counts[cat_id]
        change = f"+{after - before}" if after > before else "0"
        print(f"{cat_name:<30} {before:<12} {after:<12} {change:<12}")

def main():
    print("="*80)
    print("CREATING BALANCED COCO DATASET")
    print("="*80)

    # Create balanced datasets for each split
    balance_split('train', target_min=200)
    balance_split('valid', target_min=50)   # Lower target for validation
    balance_split('test', target_min=30)    # Lower target for test

    print(f"\n{'='*80}")
    print("✅ BALANCED DATASET CREATED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nLocation: {BALANCED_ROOT}")
    print("\nNext step: Convert to YOLO format using the balanced dataset.")

if __name__ == '__main__':
    main()
