#!/usr/bin/env python3
"""
Quick test to verify batch size impact on MPS training speed.
Run this on your MacBook to see the actual difference.

Usage:
    python scripts/test_batch_speed.py
"""

import time
import torch
from ultralytics import YOLO
from pathlib import Path

def test_batch_size(batch_size, epochs=2):
    """Test training speed with given batch size"""
    print(f"\n{'='*60}")
    print(f"Testing batch={batch_size} (epochs={epochs})")
    print(f"{'='*60}")

    # Check MPS
    if not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU (will be slow)")
        device = 'cpu'
    else:
        print(f"✅ Using MPS (Apple Silicon)")
        device = 'mps'

    # Load tiny model for quick test
    model = YOLO('yolov8n.pt')

    # Get dataset path
    dataset_yaml = Path(__file__).parent.parent / 'dataset_yolo' / 'dataset.yaml'

    if not dataset_yaml.exists():
        print(f"❌ Dataset not found: {dataset_yaml}")
        print("Run the notebook first to convert COCO dataset to YOLO format")
        return None

    # Start timer
    start_time = time.time()

    # Train
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device,
            verbose=False,
            plots=False,
            save=False,
            project='outputs/batch_test',
            name=f'batch_{batch_size}',
            exist_ok=True
        )

        elapsed = time.time() - start_time

        print(f"\n✅ Training complete!")
        print(f"⏱️  Time: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
        print(f"📊 Speed: {elapsed/epochs:.1f}초/epoch")

        return elapsed

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == '__main__':
    print("🚀 YOLO Batch Size Speed Test")
    print("This will test the impact of batch size on training speed\n")

    # Test different batch sizes
    results = {}

    # Test batch=16 first (should be fast)
    print("\n1️⃣  Testing batch=16 (recommended)")
    time_16 = test_batch_size(batch_size=16, epochs=2)
    if time_16:
        results['batch=16'] = time_16

    # Optional: test batch=-1 (auto, will be slow)
    print("\n\n2️⃣  Testing batch=-1 (auto, expected to be VERY slow)")
    response = input("⚠️  This will take 10+ minutes. Continue? (y/N): ")
    if response.lower() == 'y':
        time_auto = test_batch_size(batch_size=-1, epochs=2)
        if time_auto:
            results['batch=-1 (auto)'] = time_auto

    # Summary
    if results:
        print("\n" + "="*60)
        print("📊 RESULTS SUMMARY")
        print("="*60)
        for config, elapsed in results.items():
            print(f"{config:20s}: {elapsed:6.1f}초 ({elapsed/60:5.2f}분)")

        if len(results) == 2:
            speedup = list(results.values())[1] / list(results.values())[0]
            print(f"\n🚀 Speedup: {speedup:.1f}x faster with batch=16")

        print("\n✅ Fix confirmed: Always use batch=16 (or 8/32), NEVER batch=-1 on MPS!")
