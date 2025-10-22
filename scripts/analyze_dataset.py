#!/usr/bin/env python3
"""
Analyze COCO dataset class distribution to identify imbalance.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / 'dataset'

def analyze_split(split_name):
    """Analyze a single dataset split."""
    json_path = DATASET_ROOT / split_name / '_annotations.coco.json'

    if not json_path.exists():
        print(f"⚠️  {split_name} not found")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    # Count annotations per category
    category_counts = Counter([ann['category_id'] for ann in data['annotations']])

    # Count images per category
    images_per_category = defaultdict(set)
    for ann in data['annotations']:
        images_per_category[ann['category_id']].add(ann['image_id'])

    return {
        'total_images': len(data['images']),
        'total_annotations': len(data['annotations']),
        'category_counts': category_counts,
        'cat_id_to_name': cat_id_to_name,
        'images_per_category': {k: len(v) for k, v in images_per_category.items()}
    }

def main():
    print("="*80)
    print("COCO DATASET CLASS DISTRIBUTION ANALYSIS")
    print("="*80)

    # Analyze each split
    splits = ['train', 'valid', 'test']
    results = {}

    for split in splits:
        print(f"\n{'='*80}")
        print(f"{split.upper()} SPLIT")
        print(f"{'='*80}")

        result = analyze_split(split)
        if result is None:
            continue

        results[split] = result

        print(f"Total images: {result['total_images']}")
        print(f"Total annotations: {result['total_annotations']}")
        print(f"Avg annotations per image: {result['total_annotations'] / result['total_images']:.2f}")

        print(f"\nClass distribution (annotations):")
        print(f"{'Class':<30} {'Annotations':<15} {'Images':<10} {'%':<10}")
        print("-"*70)

        # Sort by annotation count
        sorted_cats = sorted(
            result['category_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for cat_id, count in sorted_cats:
            cat_name = result['cat_id_to_name'][cat_id]
            img_count = result['images_per_category'][cat_id]
            percentage = count / result['total_annotations'] * 100
            print(f"{cat_name:<30} {count:<15} {img_count:<10} {percentage:>6.2f}%")

    # Compute resampling requirements for train split
    if 'train' in results:
        print(f"\n{'='*80}")
        print("RESAMPLING RECOMMENDATIONS (Conservative: min 200 annotations)")
        print(f"{'='*80}\n")

        train_result = results['train']
        target_min = 200

        print(f"{'Class':<30} {'Current':<12} {'Target':<12} {'Multiplier':<12}")
        print("-"*70)

        for cat_id, count in sorted(train_result['category_counts'].items(), key=lambda x: x[1]):
            cat_name = train_result['cat_id_to_name'][cat_id]

            if count < target_min:
                multiplier = int(target_min / count) + 1
                target = count * multiplier
                print(f"{cat_name:<30} {count:<12} {target:<12} {multiplier}x")
            else:
                print(f"{cat_name:<30} {count:<12} {count:<12} 1x (no change)")

    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
