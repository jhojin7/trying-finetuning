# Troubleshooting Guide

Common issues and solutions for setting up Detectron2 on Mac Studio.

---

## Issue 1: Detectron2 Installation Fails

### Symptom
```
ERROR: Could not install packages due to an OSError
```

### Solution A: Build from source
```bash
# Clone and build detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
uv pip install -e .
cd ..
```

### Solution B: Use CPU-only wheels
```bash
# Install from pre-built wheels
python3 -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

---

## Issue 2: PyTorch MPS (GPU) Not Working

### Symptom
```
RuntimeError: MPS backend is not available
```

### Solution
Detectron2 may not support MPS on Mac. It will fall back to CPU automatically.

To verify PyTorch installation:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

**Note**: Training on CPU is slower but will still work for the demo (500 iterations).

---

## Issue 3: COCO API / pycocotools Error

### Symptom
```
ERROR: Failed building wheel for pycocotools
```

### Solution (Mac-specific)
```bash
# Install Xcode command line tools first
xcode-select --install

# Then install pycocotools
uv pip install pycocotools
```

---

## Issue 4: OpenCV Import Error

### Symptom
```
ImportError: libgthread-2.0.so.0: cannot open shared object file
```

### Solution
```bash
uv pip uninstall opencv-python
uv pip install opencv-python-headless
```

---

## Issue 5: Jupyter Kernel Not Found

### Symptom
Jupyter notebook can't find the Python kernel with installed packages.

### Solution
```bash
# Install ipykernel
uv pip install ipykernel

# Register the kernel
python3 -m ipykernel install --user --name=detectron2 --display-name="Python (detectron2)"
```

Then in Jupyter: Kernel → Change Kernel → Python (detectron2)

---

## Issue 6: Out of Memory During Training

### Symptom
```
RuntimeError: CUDA out of memory (or similar on MPS)
```

### Solution
Reduce batch size in the notebook. In cell 4, change:

```python
cfg.SOLVER.IMS_PER_BATCH = 1  # Reduce from 2 to 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # Reduce from 128
```

---

## Issue 7: Training is Too Slow

### Symptom
Training taking more than 3-4 hours for 500 iterations.

### Solutions

**A. Reduce iterations for quick demo**
```python
cfg.SOLVER.MAX_ITER = 300  # Instead of 500
```

**B. Use smaller model**
Replace in notebook cell 4:
```python
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
```

**C. Skip evaluation during training**
```python
cfg.TEST.EVAL_PERIOD = 0  # Disable evaluation during training
```

---

## Issue 8: Dataset Not Found

### Symptom
```
AssertionError: Annotation file not found
```

### Solution
Verify dataset structure:
```bash
ls dataset/train/_annotations.coco.json
ls dataset/valid/_annotations.coco.json
ls dataset/test/_annotations.coco.json
```

All three files should exist. Check paths in notebook cell 2.

---

## Issue 9: ModuleNotFoundError for detectron2

### Symptom
```
ModuleNotFoundError: No module named 'detectron2'
```

### Solution
Make sure you're using the correct Python environment:

```bash
# Check which python you're using
which python3

# Verify detectron2 is installed
python3 -c "import detectron2; print(detectron2.__version__)"
```

If using Jupyter, restart the kernel: Kernel → Restart

---

## Issue 10: Git Clone Fails for detectron2

### Symptom
```
fatal: unable to access 'https://github.com/...': Could not resolve host
```

### Solution
```bash
# Use SSH instead
uv pip install git+ssh://git@github.com/facebookresearch/detectron2.git

# Or download and install locally
wget https://github.com/facebookresearch/detectron2/archive/main.zip
unzip main.zip
cd detectron2-main
uv pip install -e .
```

---

## Still Having Issues?

### Quick Sanity Check

Run this test script:

```python
# test_installation.py
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("❌ OpenCV not installed")

try:
    import detectron2
    print(f"✅ Detectron2: {detectron2.__version__}")
except ImportError:
    print("❌ Detectron2 not installed")

try:
    from pycocotools import coco
    print("✅ pycocotools installed")
except ImportError:
    print("❌ pycocotools not installed")
```

Save and run:
```bash
python3 test_installation.py
```

All packages should show ✅

---

## Alternative: Use Docker (if all else fails)

If you're having persistent issues, use the official Detectron2 Docker image:

```bash
# Pull image
docker pull facebookresearch/detectron2:cpu

# Run container
docker run -it --rm \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  facebookresearch/detectron2:cpu \
  jupyter notebook --ip=0.0.0.0 --allow-root
```

Then access Jupyter at `http://localhost:8888`

---

## Getting Help

If you're still stuck:
1. Check Detectron2 issues: https://github.com/facebookresearch/detectron2/issues
2. Check this specific setup issue: https://github.com/facebookresearch/detectron2/issues?q=mac
3. Post a new issue with your error message and system info
