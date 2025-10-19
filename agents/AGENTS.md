# AGENTS.md

## Critical Rules

- **DO NOT READ `dataset/` FOLDERS FOR DATASETS.** It is very large and it WILL OVERFLOW your context window.
- **ALWAYS USE `uv` for python**, instead of `python` or `python3`.

## Detectron2-Specific Rules

- **DO NOT "fix" COCO annotation formats without testing first**
  - Roboflow datasets may have category_id starting from 0, this is OK
  - Detectron2 automatically maps category IDs to contiguous class indices
  - Original (0-15) and "fixed" (1-16) both work identically
  - See CLAUDE.md 2025-10-19 session for what NOT to do

- **Always set `cfg.MODEL.DEVICE = "cpu"` for Mac**
  - Mac doesn't have CUDA
  - Detectron2 doesn't fully support MPS yet
  - Will error with "Torch not compiled with CUDA enabled" otherwise

- **Test inference with multiple confidence thresholds**
  - Don't assume model is broken if no detections at threshold=0.5
  - Try 0.05, 0.1, 0.3, 0.5 to diagnose issues
  - Undertrained models produce low confidence scores (0.05-0.15 range)

- **Training iterations guidelines**
  - 500 iterations = demo/quick test only (confidence ~8-9%)
  - 2000-3000 iterations = production quality (confidence 50%+)
  - Check `fg_cls_accuracy` and `loss_cls` metrics to verify convergence