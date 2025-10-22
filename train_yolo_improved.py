#!/usr/bin/env python3
"""
Improved YOLO Training Script with Balanced Dataset

Key improvements over previous attempt:
1. Balanced dataset (min 200 annotations per class)
2. YOLOv8s model (better capacity than nano)
3. Higher learning rate (0.01 vs 0.001)
4. SGD optimizer (better for YOLO than Adam)
5. Increased classification loss weight (1.5 vs 0.5)
6. Reduced data augmentation
7. More epochs (100 vs 50)
"""

from pathlib import Path
from ultralytics import YOLO
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent
YOLO_DATASET_ROOT = PROJECT_ROOT / 'dataset_yolo_balanced'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'yolo'
dataset_yaml_path = YOLO_DATASET_ROOT / 'dataset.yaml'

# Check dataset exists
if not dataset_yaml_path.exists():
    print(f"❌ Dataset YAML not found: {dataset_yaml_path}")
    print("Please run: uv run python scripts/convert_to_yolo.py")
    exit(1)

# Device configuration
if torch.backends.mps.is_available():
    device = 'mps'
    print("✅ Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = 'cuda'
    print("✅ Using CUDA (NVIDIA GPU)")
else:
    device = 'cpu'
    print("⚠️  Using CPU (slower)")

# Load YOLOv8n model (nano - fastest)
model_size = 'yolov8n.pt'
print(f"\n🔄 Loading {model_size}...")
model = YOLO(model_size)
print(f"✅ Model loaded: {model_size}")

# IMPROVED Training Configuration
training_config = {
    'data': str(dataset_yaml_path),
    'epochs': 20,                     # Quick test: 20 epochs (~5-7 min)
    'imgsz': 640,
    'batch': 16,                      # Fixed batch size (not auto)
    'device': device,
    'project': str(OUTPUT_DIR),
    'name': 'train_balanced_quick',
    'exist_ok': True,
    'patience': 10,                   # Reduced for quick test
    'save': True,
    'plots': True,
    'verbose': True,

    # OPTIMIZER CHANGES - Key improvement!
    'optimizer': 'SGD',               # SGD > Adam for YOLO
    'lr0': 0.01,                      # 10x higher (was 0.001)
    'lrf': 0.1,                       # Higher final LR (was 0.01)
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5.0,             # Longer warmup (was 3.0)
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # LOSS WEIGHTS - Increase cls loss for better classification
    'box': 7.5,
    'cls': 1.5,                       # 3x higher (was 0.5) - KEY FIX!
    'dfl': 1.5,

    # REDUCED AUGMENTATION - Previous was too aggressive
    'hsv_h': 0.01,                    # Reduced from 0.015
    'hsv_s': 0.5,                     # Reduced from 0.7
    'hsv_v': 0.3,                     # Reduced from 0.4
    'degrees': 5.0,                   # Added slight rotation (was 0.0)
    'translate': 0.1,
    'scale': 0.3,                     # Reduced from 0.5
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 0.8,                    # Reduced (was 1.0)
    'mixup': 0.0,
    'copy_paste': 0.0,

    # Other settings
    'label_smoothing': 0.0,           # No label smoothing for now
}

print("\n" + "="*80)
print("🚀 STARTING IMPROVED YOLO TRAINING")
print("="*80)

print("\n📊 KEY IMPROVEMENTS OVER PREVIOUS ATTEMPT:")
print("  ✅ Balanced dataset (min 200 annotations/class)")
print("  ✅ Model: YOLOv8n (nano - fast)")
print("  ✅ Learning rate: 0.001 → 0.01 (10x higher)")
print("  ✅ Optimizer: Adam → SGD")
print("  ✅ Classification loss: 0.5 → 1.5 (3x higher)")
print("  ✅ Reduced augmentation (mosaic 1.0 → 0.8, scale 0.5 → 0.3)")
print("  ✅ Epochs: 50 → 20 (quick test)")

print("\n📈 EXPECTED RESULTS:")
print("  Previous attempt: mAP50 ~1-2% (failed)")
print("  Target: mAP50 >10-15% by epoch 20")
print("  Note: 20 epochs is for quick validation, not production use")

print(f"\n⏰ Training duration: ~5-7 minutes on M2")
print(f"📁 Results will be saved to: {OUTPUT_DIR / 'train_balanced_quick'}")

print("\n" + "="*80)
print("Starting training...")
print("="*80 + "\n")

# Train the model
try:
    results = model.train(**training_config)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR / 'train_balanced_quick'}")
    print(f"📊 Training plots: {OUTPUT_DIR / 'train_balanced_quick' / 'results.png'}")
    print(f"💾 Best model: {OUTPUT_DIR / 'train_balanced_quick' / 'weights' / 'best.pt'}")
    print(f"💾 Last model: {OUTPUT_DIR / 'train_balanced_quick' / 'weights' / 'last.pt'}")

    print("\n🔍 Next steps:")
    print("  1. Check results.png for training curves")
    print("  2. Look for mAP50 >10-15% (should be better than previous ~1-2%)")
    print("  3. If this works, consider training longer (50-100 epochs)")
    print("  4. Test on test set")

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    print(f"📁 Partial results saved to: {OUTPUT_DIR / 'train_balanced_quick'}")

except Exception as e:
    print(f"\n\n❌ Training failed with error:")
    print(f"   {e}")
    print(f"\n📁 Check logs at: {OUTPUT_DIR / 'train_balanced_quick'}")
    raise
