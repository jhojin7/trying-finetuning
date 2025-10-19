#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly.
Run this after setup.sh to ensure everything is working.
"""

import sys

print("="*60)
print("🧪 Testing Detectron2 Installation")
print("="*60)
print()

# Test 1: Python version
print("1. Python Version")
print(f"   Version: {sys.version}")
if sys.version_info >= (3, 7):
    print("   ✅ Python version OK (>= 3.7)")
else:
    print("   ❌ Python version too old, need >= 3.7")
print()

# Test 2: PyTorch
print("2. PyTorch")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print("   ✅ PyTorch installed")
except ImportError as e:
    print(f"   ❌ PyTorch not installed: {e}")
print()

# Test 3: OpenCV
print("3. OpenCV")
try:
    import cv2
    print(f"   Version: {cv2.__version__}")
    print("   ✅ OpenCV installed")
except ImportError as e:
    print(f"   ❌ OpenCV not installed: {e}")
print()

# Test 4: COCO API
print("4. pycocotools (COCO API)")
try:
    from pycocotools import coco
    print("   ✅ pycocotools installed")
except ImportError as e:
    print(f"   ❌ pycocotools not installed: {e}")
print()

# Test 5: Detectron2
print("5. Detectron2")
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    print(f"   Version: {detectron2.__version__}")
    print("   ✅ Detectron2 installed and importable")
except ImportError as e:
    print(f"   ❌ Detectron2 not installed: {e}")
    print("   💡 Try: uv pip install 'git+https://github.com/facebookresearch/detectron2.git'")
print()

# Test 6: Matplotlib
print("6. Matplotlib")
try:
    import matplotlib
    print(f"   Version: {matplotlib.__version__}")
    print("   ✅ Matplotlib installed")
except ImportError as e:
    print(f"   ❌ Matplotlib not installed: {e}")
print()

# Test 7: Jupyter
print("7. Jupyter Notebook")
try:
    import notebook
    import jupyter
    print("   ✅ Jupyter installed")
except ImportError as e:
    print(f"   ❌ Jupyter not installed: {e}")
print()

# Test 8: Dataset files
print("8. Dataset Files")
import os
dataset_files = [
    "dataset/train/_annotations.coco.json",
    "dataset/valid/_annotations.coco.json",
    "dataset/test/_annotations.coco.json"
]
all_exist = True
for filepath in dataset_files:
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"   {status} {filepath}")
    all_exist = all_exist and exists

if all_exist:
    print("   ✅ All dataset files found")
else:
    print("   ❌ Some dataset files missing")
print()

# Test 9: Output directories
print("9. Output Directories")
output_dirs = [
    "notebooks",
    "outputs/models",
    "outputs/predictions",
    "scripts"
]
for dirpath in output_dirs:
    exists = os.path.exists(dirpath)
    status = "✅" if exists else "❌"
    print(f"   {status} {dirpath}")
print()

# Test 10: Quick Detectron2 functional test
print("10. Detectron2 Functional Test")
try:
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    print("   ✅ Can load Detectron2 config")
except Exception as e:
    print(f"   ❌ Detectron2 config test failed: {e}")
print()

print("="*60)
print("🎯 Installation Test Complete")
print("="*60)
print()
print("Next steps:")
print("1. If all tests passed (✅), run: jupyter notebook")
print("2. Open: notebooks/detectron2_finetuning.ipynb")
print("3. Start training!")
print()
print("If any tests failed (❌), check TROUBLESHOOTING.md")
print()
