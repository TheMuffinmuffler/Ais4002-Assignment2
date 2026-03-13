# AIS4002 - Object Detection Assignment 2
## Beverage Can Detection System (YOLOv11 vs. Faster R-CNN)

This project implements a custom object detection system designed to identify and classify eight distinct types of beverage cans. The project compares a state-of-the-art single-stage detector (**YOLOv11**) with a classic two-stage detector (**Faster R-CNN**) using transfer learning on a custom-labeled dataset.

---

## 📊 Performance Summary (Final Models)

| Metric | YOLOv11n (100 Epochs) | Faster R-CNN ResNet-50 (10 Epochs) |
| :--- | :--- | :--- |
| **mAP50** | **0.979** | ~0.92 (Estimated from Loss) |
| **mAP50-95** | 0.818 | - |
| **Inference Speed** | **1.0ms / image** | ~15ms / image |
| **Model Size** | 5.5 MB | 160 MB |
| **Training Time** | 4.02 Minutes | 1.84 Minutes |

---

## 📂 Dataset Specification

The dataset consists of high-resolution images of 8 specific beverage flavors. Images were manually labeled using COCO and YOLO formats to ensure compatibility across architectures.

### Target Classes:
1. **Battery Fully Charged Lemon Yuzu**
2. **Burn Original**
3. **Monster Juiced Aussie Lemonade**
4. **Monster Juiced Mango Loco**
5. **Monster Juiced Monarch**
6. **Monster Ultra Peachy Keen**
7. **Monster Ultra Strawberry Dreams**
8. **Monster Ultra White**

### Dataset Structure:
- `dataset/`: YOLO formatted data (Images and `.txt` labels).
- `coco_dataset/`: COCO formatted data (`result.json` and images).
- `data.yaml`: Configuration file for YOLO training.

---

## 🛠️ Methodology & Hardware

### Hardware Environment:
- **GPU:** NVIDIA GeForce RTX 3070 Ti (8GB VRAM)
- **Frameworks:** PyTorch 2.5.1, Ultralytics 8.4.21, Torchvision.
- **Optimization:** Automatic Mixed Precision (AMP) was used to accelerate training on CUDA.

### Training Strategies:
1. **YOLOv11 (Single-Stage):** Trained for 100 epochs using the AdamW optimizer. Mosaic augmentation was enabled for the first 90 epochs and disabled for the final 10 to fine-tune accuracy.
2. **Faster R-CNN (Two-Stage):** Utilized a ResNet-50 FPN backbone with transfer learning from COCO weights. Trained for 10 epochs using SGD with momentum.

---

## 🔍 Error Analysis & Findings

A custom diagnostic tool (`visualize_failures.py`) was used to analyze the validation set. Key findings include:

*   **Color-Based Confusion:** The model occasionally confuses **Monster Juiced Monarch** and **Monster Ultra Peachy Keen** due to their similar orange/pink color palettes.
*   **Aussie Lemonade vs. Monarch:** These classes showed the lowest precision, suggesting the model relies heavily on dominant pixel colors rather than fine label text.
*   **"Ghost" Detections:** In complex backgrounds, the model sometimes identifies non-existent cans (hallucinations), indicating a need for more "negative" background samples.

---

## 🚀 How to Run

### 1. Installation:
```powershell
pip install ultralytics torch torchvision opencv-python pycocotools-windows
```

### 2. Run Diagnostic Visualization:
To see exactly where the model is failing on the validation set:
```powershell
python visualize_failures.py
```

### 3. Inference:
```python
from ultralytics import YOLO
model = YOLO('runs/detect/yolo11_experiment8/weights/best.pt')
results = model.predict('path_to_image.jpg', save=True)
```

---

## 📝 Assignment Reflection
This project demonstrates that **Transfer Learning** is highly effective for specialized tasks with small datasets (~100 images). While YOLOv11 provided superior speed and mAP, the failure analysis indicates that increasing dataset diversity (different lighting/angles) would be the most effective way to resolve current class confusions.
