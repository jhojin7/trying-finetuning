# Quick Start Guide - Sunday Demo Prep

**Goal**: Get a working demo ready for tomorrow (Monday)
**Time available**: ~2 hours hands-on + training time

---

## Step 1: Install Dependencies (15-20 min)

Run the setup script:

```bash
./setup.sh
```

This will install:
- PyTorch with MPS support (Mac GPU acceleration)
- Detectron2
- Dependencies (opencv, matplotlib, pycocotools, jupyter)

**Troubleshooting**: If detectron2 installation fails, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

---

## Step 2: Start Jupyter Notebook (5 min)

```bash
jupyter notebook
```

This will open Jupyter in your browser.

---

## Step 3: Run Fine-tuning Notebook (1-2 hours)

1. Open `notebooks/detectron2_finetuning.ipynb`
2. Run all cells in order (Shift+Enter)
3. The training will run for ~500 iterations
   - On Mac Studio, this takes 1-2 hours
   - You can work on other things while it trains
   - Watch for decreasing loss values (good sign!)

**What happens**:
- Loads electronic components dataset in COCO format
- Registers train/val/test splits
- Fine-tunes Faster R-CNN model (pre-trained on COCO)
- Saves model to `outputs/models/model_final.pth`
- Creates visualizations in `outputs/predictions/`

---

## Step 4: Verify Results (5 min)

After training completes, check:

1. **Model saved**: `outputs/models/model_final.pth` exists
2. **Visualizations created**:
   - `outputs/predictions/training_samples.png` - shows training data
   - `outputs/predictions/test_predictions.png` - shows model predictions
   - `outputs/predictions/before_after_comparison.png` - **KEY FOR DEMO!**

The before/after comparison shows the improvement from fine-tuning!

---

## Step 5: Optional - Batch Pre-labeling (10 min)

Test the inference script on new images:

```bash
python scripts/predict.py \
    --input dataset/test \
    --output outputs/predictions \
    --visualize
```

This generates:
- `predictions_coco.json` - COCO format predictions for labeling tools
- `prediction_summary.json` - Statistics
- `visualizations/` - Annotated images

---

## Demo Talking Points (for tomorrow)

### 1. Problem Statement
- Manual labeling is time-consuming
- Want to implement pre-labeling workflow
- Human labelers focus on corrections, not full annotation

### 2. Solution
- Fine-tuned Detectron2 on electronic components dataset
- Transfer learning from COCO pre-trained model
- Quick training: 500 iterations in 1-2 hours

### 3. Key Visuals to Show

**A. Training Samples** (`training_samples.png`)
- Shows the dataset we're working with
- Electronic components with bounding boxes

**B. Before/After Comparison** (`before_after_comparison.png`) ⭐
- **MOST IMPORTANT SLIDE**
- Left: Pre-trained COCO model (poor results on electronics)
- Right: Fine-tuned model (much better!)
- Shows clear improvement

**C. Test Predictions** (`test_predictions.png`)
- Model working on unseen test images
- Shows it can generalize to new data

### 4. Pre-labeling Workflow Demo

Show the inference script:
```bash
python scripts/predict.py --input /new/images --output /predictions --visualize
```

Explain:
- Batch process: run on 100s-1000s of images
- Outputs COCO format (compatible with CVAT, FiftyOne, Roboflow)
- Humans review and fix mistakes
- 80% time savings vs full manual labeling

### 5. Next Steps
- Train longer (2000+ iterations) for better accuracy
- Integrate with CVAT for review workflow
- Set up active learning pipeline
- Monitor model performance over time

---

## Time Budget Recap

| Task | Time |
|------|------|
| Install dependencies | 15-20 min |
| Setup notebook | 5 min |
| **Training** | **1-2 hours** (can multitask) |
| Verify results | 5 min |
| Test inference | 10 min |
| **Total hands-on** | **~40 min + 1-2h training** |

---

## Backup Plan

If training doesn't finish in time:
1. You can demo the notebook showing the training setup
2. Use the visualizations from earlier cells (training samples)
3. Explain the workflow even without the final model

The notebook is well-documented and can serve as a standalone demo!

---

## Questions?

Check these files:
- `agents/plan.md` - Full project plan
- `README.md` - Project overview
- Notebook has extensive comments and explanations

---

**🚀 You got this! The hard work is done, now just run the notebook and let it train.**
