#!/usr/bin/env python3
"""
Standalone training script extracted from detectron2_finetuning.ipynb
Runs training on the synthetic dataset.
"""

import os
import json
import sys

# Detectron2 imports
try:
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()

    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
except ImportError:
    print("ERROR: detectron2 not installed")
    print("Install with: uv pip install --system 'git+https://github.com/facebookresearch/detectron2.git'")
    sys.exit(1)

print(f"Detectron2 version: {detectron2.__version__}")
print(f"Working directory: {os.getcwd()}")

# Dataset paths
DATA_ROOT = "dataset"
TRAIN_JSON = os.path.join(DATA_ROOT, "train/_annotations.coco.json")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_JSON = os.path.join(DATA_ROOT, "valid/_annotations.coco.json")
VAL_DIR = os.path.join(DATA_ROOT, "valid")

# Register datasets
print("\n📊 Registering datasets...")
register_coco_instances("synthetic_train", {}, TRAIN_JSON, TRAIN_DIR)
register_coco_instances("synthetic_val", {}, VAL_JSON, VAL_DIR)

# Get metadata
metadata = MetadataCatalog.get("synthetic_train")

print(f"  Training images: {len(list(open(TRAIN_JSON).read().count('\"id\"')//3))} images")
print(f"  Categories: {metadata.thing_classes}")
print(f"  Number of categories: {len(metadata.thing_classes)}")

# Configure model
print("\n⚙️  Configuring model...")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("synthetic_train",)
cfg.DATASETS.TEST = ("synthetic_val",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300  # Quick test - 300 iterations
cfg.SOLVER.STEPS = (200,)
cfg.SOLVER.CHECKPOINT_PERIOD = 150

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)

cfg.OUTPUT_DIR = "outputs/models"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.DEVICE = "cpu"

print(f"  Model: Faster R-CNN R50-FPN")
print(f"  Device: CPU")
print(f"  Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
print(f"  Output: {cfg.OUTPUT_DIR}")

# Train
print(f"\n🚀 Starting training for {cfg.SOLVER.MAX_ITER} iterations...")
print("=" * 60)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("=" * 60)
print(f"✅ Training complete!")
print(f"📁 Model saved to: {cfg.OUTPUT_DIR}/model_final.pth")
