#!/usr/bin/env python3
"""
Batch Pre-labeling Script for Electronic Components Detection

This script runs inference on a batch of images using the fine-tuned Detectron2 model
and generates predictions in COCO format for human review.

Usage:
    python scripts/predict.py --input /path/to/images --output /path/to/output
    python scripts/predict.py --input dataset/test --output outputs/predictions --visualize
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def setup_cfg(model_path, confidence_threshold=0.5, num_classes=None):
    """
    Configure Detectron2 for inference.

    Args:
        model_path: Path to trained model weights
        confidence_threshold: Minimum confidence score for detections
        num_classes: Number of object classes (optional)

    Returns:
        cfg: Detectron2 config object
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for Mac

    if num_classes is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    return cfg


def get_image_files(input_dir):
    """Get all image files from input directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    input_path = Path(input_dir)
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))

    return sorted(image_files)


def predict_batch(predictor, image_files, metadata, output_dir, visualize=True):
    """
    Run predictions on a batch of images.

    Args:
        predictor: Detectron2 predictor
        image_files: List of image file paths
        metadata: Dataset metadata for visualization
        output_dir: Directory to save outputs
        visualize: Whether to save visualization images

    Returns:
        predictions: List of prediction dictionaries
    """
    predictions = []
    vis_dir = os.path.join(output_dir, "visualizations")

    if visualize:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"\n🔍 Running inference on {len(image_files)} images...")

    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Run inference
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        # Prepare prediction data
        pred_data = {
            "file_name": img_path.name,
            "image_path": str(img_path),
            "image_size": img.shape[:2],  # (height, width)
            "num_detections": len(instances),
            "detections": []
        }

        # Extract detection information
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        scores = instances.scores.numpy() if instances.has("scores") else []
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []

        for i in range(len(instances)):
            x1, y1, x2, y2 = boxes[i]
            detection = {
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # COCO format: [x, y, width, height]
                "category_id": int(classes[i]),
                "category_name": metadata.thing_classes[classes[i]] if metadata else str(classes[i]),
                "score": float(scores[i])
            }
            pred_data["detections"].append(detection)

        predictions.append(pred_data)

        # Save visualization
        if visualize:
            v = Visualizer(img[:, :, ::-1],
                          metadata=metadata,
                          scale=1.0,
                          instance_mode=ColorMode.IMAGE)
            out = v.draw_instance_predictions(instances)

            vis_path = os.path.join(vis_dir, f"pred_{img_path.stem}.jpg")
            cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])

    return predictions


def save_predictions_coco(predictions, output_path, categories):
    """
    Save predictions in COCO format for easy import into labeling tools.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save JSON file
        categories: List of category dictionaries
    """
    coco_format = {
        "info": {
            "description": "Pre-labeled Electronic Components Detection",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        },
        "images": [],
        "annotations": [],
        "categories": categories
    }

    annotation_id = 1

    for img_id, pred in enumerate(predictions, start=1):
        # Add image info
        height, width = pred["image_size"]
        image_info = {
            "id": img_id,
            "file_name": pred["file_name"],
            "height": int(height),
            "width": int(width)
        }
        coco_format["images"].append(image_info)

        # Add annotations
        for det in pred["detections"]:
            annotation = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": det["category_id"],
                "bbox": det["bbox"],
                "area": float(det["bbox"][2] * det["bbox"][3]),
                "iscrowd": 0,
                "score": det["score"]
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"\n✅ Saved COCO predictions to: {output_path}")


def generate_summary(predictions, output_dir):
    """Generate and save prediction summary statistics."""
    total_images = len(predictions)
    total_detections = sum(p["num_detections"] for p in predictions)
    avg_detections = total_detections / total_images if total_images > 0 else 0

    # Count detections per category
    category_counts = {}
    all_scores = []

    for pred in predictions:
        for det in pred["detections"]:
            cat_name = det["category_name"]
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            all_scores.append(det["score"])

    avg_confidence = np.mean(all_scores) if all_scores else 0

    summary = {
        "summary_statistics": {
            "total_images": total_images,
            "total_detections": total_detections,
            "average_detections_per_image": round(avg_detections, 2),
            "average_confidence": round(avg_confidence, 4)
        },
        "detections_per_category": category_counts,
        "images_with_predictions": [
            {
                "file_name": p["file_name"],
                "num_detections": p["num_detections"]
            }
            for p in predictions
        ]
    }

    summary_path = os.path.join(output_dir, "prediction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Prediction Summary:")
    print(f"   Total images processed: {total_images}")
    print(f"   Total detections: {total_detections}")
    print(f"   Average detections per image: {avg_detections:.2f}")
    print(f"   Average confidence: {avg_confidence:.4f}")
    print(f"   Detections by category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {cat}: {count}")

    print(f"\n✅ Saved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch pre-labeling with Detectron2")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for predictions")
    parser.add_argument("--model", type=str, default="../outputs/models/model_final.pth",
                       help="Path to trained model weights")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for predictions (0-1)")
    parser.add_argument("--visualize", action="store_true",
                       help="Save visualization images")
    parser.add_argument("--dataset-name", type=str, default="electronics_train",
                       help="Name of registered dataset for metadata")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print("🔮 Detectron2 Batch Pre-labeling")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Visualize: {args.visualize}")

    # Read categories from COCO JSON
    DATA_ROOT = "dataset"
    TRAIN_JSON = os.path.join(DATA_ROOT, "train/_annotations.coco.json")
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")

    with open(TRAIN_JSON, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']  # Original category IDs (1-16)

    # Extract class names in order of category ID
    # COCO categories are sorted by ID, Detectron2 expects thing_classes in ID order
    thing_classes = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    num_classes = len(thing_classes)

    # Register dataset with proper metadata
    if args.dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(args.dataset_name)
        MetadataCatalog.remove(args.dataset_name)

    register_coco_instances(args.dataset_name, {}, TRAIN_JSON, TRAIN_DIR)
    metadata = MetadataCatalog.get(args.dataset_name)

    # Manually set thing_classes if not automatically populated
    if not hasattr(metadata, 'thing_classes') or len(metadata.thing_classes) == 0:
        metadata.thing_classes = thing_classes

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {thing_classes}")
    print(f"Category IDs: {[c['id'] for c in categories]}")

    # Setup configuration
    cfg = setup_cfg(args.model, args.confidence, num_classes=num_classes)
    predictor = DefaultPredictor(cfg)

    # Get image files
    image_files = get_image_files(args.input)
    if not image_files:
        print(f"❌ No images found in {args.input}")
        return

    # Run predictions
    predictions = predict_batch(predictor, image_files, metadata, args.output, args.visualize)

    # Save predictions in COCO format
    coco_output_path = os.path.join(args.output, "predictions_coco.json")
    save_predictions_coco(predictions, coco_output_path, categories)

    # Generate summary
    generate_summary(predictions, args.output)

    print("\n" + "="*60)
    print("✅ Batch pre-labeling complete!")
    print("="*60)
    print(f"\nOutputs saved to: {args.output}")
    print(f"  - COCO predictions: predictions_coco.json")
    print(f"  - Summary: prediction_summary.json")
    if args.visualize:
        print(f"  - Visualizations: visualizations/")
    print("\n💡 Next: Import predictions_coco.json into your labeling tool for human review")


if __name__ == "__main__":
    main()
