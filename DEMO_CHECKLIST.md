# Demo Checklist - Sunday to Monday

Quick checklist to get your demo ready for tomorrow.

---

## ⏰ Timeline

**Today (Sunday, noon)**:
- [ ] Install dependencies (~20 min)
- [ ] Verify installation (~5 min)
- [ ] Start training (~5 min to start, then 1-2 hours to run)
- [ ] Verify outputs (~5 min)

**Tomorrow (Monday)**:
- [ ] Prepare demo talking points
- [ ] Show visualizations
- [ ] Run inference demo (optional)

---

## 📋 Step-by-Step Checklist

### Part 1: Setup (30 minutes)

- [ ] **Step 1.1**: Run setup script
  ```bash
  cd /Users/hojinjang/work/trying-finetuning
  ./setup.sh
  ```
  Expected: All packages install successfully with uv

- [ ] **Step 1.2**: Verify installation
  ```bash
  python3 test_installation.py
  ```
  Expected: All tests show ✅

- [ ] **Step 1.3**: Fix any issues
  - If anything fails, check `TROUBLESHOOTING.md`
  - Most common issue: detectron2 installation
  - Solution: Build from source (see troubleshooting guide)

---

### Part 2: Training (1-2 hours)

- [ ] **Step 2.1**: Start Jupyter
  ```bash
  jupyter notebook
  ```
  Browser should open automatically

- [ ] **Step 2.2**: Open notebook
  - Navigate to `notebooks/detectron2_finetuning.ipynb`
  - Click to open

- [ ] **Step 2.3**: Run all cells
  - Click: Kernel → Restart & Run All
  - Or: Run each cell with Shift+Enter
  - **This will take 1-2 hours to complete**

- [ ] **Step 2.4**: Monitor progress
  - Watch for decreasing loss values
  - Loss should go from ~1.0 → ~0.3-0.5
  - No need to watch constantly - do other work!

- [ ] **Step 2.5**: Check for completion
  - Final cell should print "🎉 DEMO READY FOR TOMORROW!"
  - Model saved to `outputs/models/model_final.pth`

---

### Part 3: Verify Results (10 minutes)

- [ ] **Step 3.1**: Check model file exists
  ```bash
  ls -lh outputs/models/model_final.pth
  ```
  Expected: File exists, ~150-200MB size

- [ ] **Step 3.2**: Check visualizations created
  ```bash
  ls outputs/predictions/
  ```
  Expected files:
  - `training_samples.png`
  - `test_predictions.png`
  - `before_after_comparison.png` ← **MOST IMPORTANT**

- [ ] **Step 3.3**: View visualizations
  ```bash
  open outputs/predictions/before_after_comparison.png
  ```
  Expected: Side-by-side comparison showing improvement

- [ ] **Step 3.4**: Check evaluation metrics
  - Look at notebook output for COCO metrics
  - Note the AP (Average Precision) values
  - These show model performance

---

### Part 4: Optional - Test Inference (10 minutes)

- [ ] **Step 4.1**: Run batch prediction
  ```bash
  python scripts/predict.py \
      --input dataset/test \
      --output outputs/predictions/batch_test \
      --visualize
  ```

- [ ] **Step 4.2**: Check outputs
  ```bash
  ls outputs/predictions/batch_test/
  ```
  Expected:
  - `predictions_coco.json` - predictions in COCO format
  - `prediction_summary.json` - statistics
  - `visualizations/` - annotated images

---

## 🎥 Demo Preparation (Do tonight or tomorrow morning)

- [ ] **Prepare talking points**
  - Read: "Demo Talking Points" section in `QUICKSTART.md`
  - Key message: AI pre-labels, humans fix → 80% time savings

- [ ] **Key visuals to show** (in order):
  1. Training samples (`training_samples.png`)
     - "This is our dataset - electronic components"
  2. Before/After comparison (`before_after_comparison.png`)
     - "Left: pre-trained model (bad), Right: fine-tuned (good)"
     - **Most impactful slide!**
  3. Test predictions (`test_predictions.png`)
     - "Model working on new images"

- [ ] **Live demo** (optional, if confident):
  - Show the Jupyter notebook
  - Show running inference script
  - Show COCO JSON output

- [ ] **Explain workflow**:
  ```
  New images → Batch pre-label → Export COCO JSON →
  Import to CVAT → Humans review/fix → Export final labels
  ```

---

## ✅ Success Criteria

You're ready for the demo if you have:

- ✅ Model file: `outputs/models/model_final.pth`
- ✅ Before/after comparison image
- ✅ Understanding of the workflow
- ✅ Ability to explain time savings

Even if training isn't perfect, you can show:
- The dataset and problem
- The approach (transfer learning)
- The workflow (batch pre-labeling)
- The tooling (notebook + script)

---

## 🆘 If Something Goes Wrong

### Training is taking too long
- Reduce iterations in notebook: `cfg.SOLVER.MAX_ITER = 300`
- Skip evaluation: `cfg.TEST.EVAL_PERIOD = 0`
- Still valuable to demo the setup even without final model

### Installation issues
- Check `TROUBLESHOOTING.md`
- Most issues have solutions there
- Can demo with mock outputs if needed

### Out of time
**Minimum viable demo**:
1. Show the dataset structure
2. Show the notebook code (even if not run)
3. Explain the workflow
4. Show what outputs would be created

This still demonstrates:
- Understanding of the problem
- Technical approach
- Clear path to implementation

---

## 📞 Quick Reference

- Full plan: `agents/plan.md`
- Quick start: `QUICKSTART.md`
- Troubleshooting: `TROUBLESHOOTING.md`
- Main README: `README.md`

---

**Time check**: It's Sunday noon. Aim to start training by 2pm, so it finishes by 4pm. That leaves evening to review and prepare!

**You got this! 💪**
