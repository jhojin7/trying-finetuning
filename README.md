# Detectron2 Fine-tuning for Pre-labeling Workflow

Quick demo project for fine-tuning an object detection model on electronic components dataset.

**Dataset**: [Electronic Components Detection](https://universe.roboflow.com/cv2024-4skvf/electronic-components-detection-jzf42)
**Model**: [Detectron2](https://github.com/facebookresearch/detectron2) (Faster R-CNN)

---

## 🎯 Project Goals

- Fine-tune object detection model for pre-labeling workflow
- Enable batch pre-labeling on new images
- Human labelers focus on corrections rather than full annotation
- Run everything locally on Mac Studio

---

## 🚀 Quick Start (For Demo Tomorrow)

**Read this first**: [QUICKSTART.md](./QUICKSTART.md)

### 1. Install dependencies
```bash
./setup.sh
```

### 2. Test installation
```bash
python3 test_installation.py
```

### 3. Start training
```bash
jupyter notebook
# Open notebooks/detectron2_finetuning.ipynb
# Run all cells
```

### 4. Run batch pre-labeling
```bash
python scripts/predict.py \
    --input dataset/test \
    --output outputs/predictions \
    --visualize
```

**Total time**: ~1.5 hours hands-on + 1-2 hours training

---

## 📁 Directory Structure

```
.
├── README.md                      # This file
├── QUICKSTART.md                  # Quick start guide for demo
├── TROUBLESHOOTING.md             # Common issues and solutions
├── setup.sh                       # Installation script (uses uv)
├── requirements.txt               # Python dependencies
├── test_installation.py           # Verify installation
│
├── agents/                        # Project planning
│   └── plan.md                    # Detailed implementation plan
│
├── dataset/                       # COCO format dataset
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg
│   └── test/
│       ├── _annotations.coco.json
│       └── *.jpg
│
├── notebooks/
│   └── detectron2_finetuning.ipynb  # Main training notebook
│
├── scripts/
│   └── predict.py                   # Batch pre-labeling script
│
└── outputs/
    ├── models/                      # Saved model checkpoints
    │   └── model_final.pth
    └── predictions/                 # Visualizations and results
        ├── training_samples.png
        ├── test_predictions.png
        ├── before_after_comparison.png
        ├── predictions_coco.json
        └── prediction_summary.json
```

---

## 📊 What Gets Created

After running the notebook, you'll have:

1. **Fine-tuned model**: `outputs/models/model_final.pth`
2. **Visualizations**:
   - Training samples with ground truth
   - Test predictions from fine-tuned model
   - Before/after comparison (pre-trained vs fine-tuned)
3. **Evaluation metrics**: COCO AP, AR scores
4. **Pre-labeling outputs**: COCO format JSON for import into labeling tools

---

## 🛠️ Tech Stack

- **Framework**: Detectron2 (Facebook Research)
- **Model**: Faster R-CNN with ResNet-50 FPN backbone
- **Dataset format**: COCO JSON
- **Package manager**: uv (fast Python package installer)
- **Platform**: Mac Studio (Apple Silicon)

---

## 💡 Project Notes

- **Batch processing**: Not real-time; process images in batches
- **Human-in-the-loop**: Humans fix model mistakes, not full annotation
- **Future capabilities**: Object detection + instance segmentation
- **Low maintenance**: Minimal manual intervention required
- **Local execution**: Everything runs on Mac Studio

### Recommended Tools for Production
- **CVAT**: Open-source annotation tool with pre-labeling support
- **FiftyOne**: Dataset management and visualization
- **Cleanlab**: Data quality and label error detection
- **Roboflow**: Dataset versioning and augmentation

---

## 📈 Training Configuration

**Quick demo settings** (500 iterations, 1-2 hours):
- Base learning rate: 0.00025
- Batch size: 2 images per batch
- ROI batch size: 128
- Checkpoints: Every 250 iterations

**Production settings** (in notebook comments):
- Increase to 2000-3000 iterations
- Tune learning rate schedule
- Add data augmentation
- Enable evaluation during training

---

## 🎥 Demo Talking Points

1. **Problem**: Manual labeling is time-consuming
2. **Solution**: Pre-label with AI, humans fix mistakes
3. **Visual**: Show before/after comparison
4. **Workflow**: Batch process → COCO JSON → Import to CVAT → Human review
5. **Results**: 80%+ time savings vs full manual labeling

**Key visual**: `outputs/predictions/before_after_comparison.png`

---

## 📚 Resources

### Detectron2
- [Official Tutorial](https://blog.roboflow.com/how-to-train-detectron2/)
- [Roboflow Notebooks](https://github.com/roboflow/notebooks)
- [Korean Tutorial](https://kkiho.tistory.com/58)

### Data Quality
- [Cleanlab Object Detection](https://docs.cleanlab.ai/stable/tutorials/object_detection.html)

### Labeling Tools
- [CVAT](https://github.com/opencv/cvat) - Open-source annotation
- [FiftyOne](https://github.com/voxel51/fiftyone) - Dataset visualization

---

## 🐛 Troubleshooting

Having issues? Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common problems and solutions.

---

## ✅ Next Steps After Demo

1. **Longer training**: Increase iterations to 2000+
2. **Hyperparameter tuning**: Learning rate, batch size, augmentation
3. **Active learning**: Prioritize uncertain predictions for human review
4. **CVAT integration**: Set up review workflow
5. **Model monitoring**: Track performance over time
6. **Instance segmentation**: Add mask prediction capability

---

## 📝 License

This is a demo project. Dataset from Roboflow Universe. Detectron2 is licensed under Apache 2.0.