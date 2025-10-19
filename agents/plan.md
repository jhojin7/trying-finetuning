# Quick Detectron2 Demo Plan (for tomorrow)

## Overview
Creating a quick demo for object detection + pre-labeling workflow using detectron2 on electronic components dataset.

**Timeline**: Sunday afternoon → Monday demo
**Time Budget**: 1-2 hours hands-on + training time
**Demo Goals**: Show fine-tuned model working, pre-labeling on new images, visual results

---

## Phase 1: Environment Setup (15-20 min)
1. **Install PyTorch with MPS support** for Mac Studio (Apple Silicon)
   - Use conda or pip to install PyTorch that supports Metal Performance Shaders
2. **Install detectron2**
   - Build from source for Mac compatibility (or use pre-built wheel if available)
   - Install dependencies: opencv-python, matplotlib, pycocotools
3. **Create project structure**
   - Set up notebooks/ and outputs/ directories as per README

## Phase 2: Create Fine-tuning Notebook (30 min setup)
1. **Create `notebooks/detectron2_finetuning.ipynb`** with:
   - COCO dataset registration for your electronic components
   - Load pre-trained Faster R-CNN or RetinaNet model (COCO weights)
   - Configure for transfer learning with reduced epochs (~300-500 iterations for quick demo)
   - Training loop with periodic evaluation
   - Model saving to `outputs/models/`

## Phase 3: Create Inference Script (20 min)
1. **Create `scripts/predict.py`** for batch pre-labeling:
   - Load fine-tuned model
   - Run inference on test set images
   - Generate visualizations with bounding boxes + class labels
   - Save predictions in COCO format for human review
   - Save visual outputs to `outputs/predictions/`

## Phase 4: Training & Validation (1-2 hours)
1. **Run the training notebook**
   - Quick fine-tuning session (~300-500 iterations)
   - Monitor loss curves
2. **Run inference on test set** to generate demo outputs

## Phase 5: Demo Materials (10 min)
1. **Create simple comparison notebook** showing:
   - Before: pre-trained COCO model on your images (poor results)
   - After: fine-tuned model on your images (improved results)
   - Side-by-side visualization for impact

---

## Key Optimizations for Speed
- Use Faster R-CNN R50-FPN (smaller, faster than R101)
- Reduce training iterations (300-500 vs typical 1000+)
- Use MPS backend on Mac Studio for GPU acceleration
- Pre-download model weights to save time

## Total Estimated Time
**Hands-on time**: ~1.5 hours + training time (can run while doing other things)
