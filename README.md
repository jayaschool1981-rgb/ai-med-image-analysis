# AI-Powered Medical Image Analysis

Detect Pneumonia vs Normal from Chest X-ray images using EfficientNetB0 + Grad-CAM explainability.

> ⚠️ **Disclaimer:** This is a research-learning project. Not approved for diagnostic or clinical use.

---

## 🚀 Features

| Feature | Status |
|--------|--------|
| Deep learning classifier (EfficientNetB0) | ✅ |
| Class imbalance handling (class weights) | ✅ |
| Data augmentation (safe medical augmentations) | ✅ |
| Evaluation: ROC-AUC, PR-AUC, Confusion Matrix | ✅ |
| Grad-CAM explainability | ✅ |
| REST API with FastAPI | ✅ |
| Minimal React UI + image upload | ✅ |

---

## 📂 Folder Structure

/.
├─ data/
│ ├─ raw/
│ ├─ processed/
│ └─ samples/
├─ scripts/
│ ├─ generate_dummy_data.py
│ ├─ prepare_data.py
│ └─ train.py
├─ notebooks/
│ └─ training.ipynb
├─ api/
│ ├─ app.py
│ └─ utils/
│ └─ inference.py
├─ models/
│ └─ v1/
├─ frontend/
│ ├─ src/
│ │ ├─ App.jsx
│ │ └─ main.jsx
│ ├─ index.html
│ ├─ vite.config.js
│ └─ package.json
├─ requirements.txt
└─ README.md

---

## 📊 Dataset

### Public dataset options

| Dataset | Link |
|--------|------|
| Kaggle - Chest X-Ray Images (Pneumonia) | https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia |
| NIH ChestX-ray14 | https://nihcc.app.box.com/v/ChestXray-NIHCC |

Put your dataset under:


OR generate synthetic dummy sample data:

```bash
python scripts/generate_dummy_data.py
