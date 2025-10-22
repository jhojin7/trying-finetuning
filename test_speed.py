#!/usr/bin/env python3
"""Quick YOLO training speed test - CPU vs MPS"""

import time
from pathlib import Path
from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path(__file__).parent
DATASET_YAML = PROJECT_ROOT / 'dataset_yolo' / 'dataset.yaml'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'yolo_speed_test'

# Test configuration
TEST_CONFIG = {
    'data': str(DATASET_YAML),
    'epochs': 2,
    'imgsz': 640,
    'batch': 64,
    'workers': 8,
    'cache': True,
    'amp': False,
    'save': False,
    'plots': False,
    'verbose': False,
    'project': str(OUTPUT_DIR),
    'exist_ok': True,
}

def test_device(device_name):
    """Test training speed on specified device"""
    print(f"\n{'='*60}")
    print(f"Testing: {device_name.upper()}")
    print('='*60)

    config = TEST_CONFIG.copy()
    config['device'] = device_name
    config['name'] = f'test_{device_name}'

    model = YOLO('yolov8n.pt')

    start_time = time.time()
    try:
        results = model.train(**config)
        elapsed = time.time() - start_time

        print(f"\n✅ {device_name.upper()} completed in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
        return elapsed
    except Exception as e:
        print(f"\n❌ {device_name.upper()} failed: {e}")
        return None

if __name__ == '__main__':
    print("YOLO Training Speed Test")
    print(f"Dataset: {DATASET_YAML}")
    print(f"Config: epochs=2, batch=64, cache=True, workers=8")

    # Test CPU
    cpu_time = test_device('cpu')

    # Test MPS
    mps_time = test_device('mps')

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print('='*60)
    if cpu_time:
        print(f"CPU: {cpu_time:.1f}s ({cpu_time/60:.2f} min)")
    if mps_time:
        print(f"MPS: {mps_time:.1f}s ({mps_time/60:.2f} min)")
    if cpu_time and mps_time:
        faster = 'CPU' if cpu_time < mps_time else 'MPS'
        ratio = max(cpu_time, mps_time) / min(cpu_time, mps_time)
        print(f"\n{faster} is {ratio:.1f}x faster")
    print('='*60)
