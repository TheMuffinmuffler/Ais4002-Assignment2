# Ais4002 Assignment 2: Beverage Detection Comparison

This project compares two major object detection architectures: **YOLO11 (Nano)** and **Faster R-CNN (ResNet-50)**. Both models are trained on a custom dataset of beverage cans/bottles.

## 🚀 Project Overview
The goal is to evaluate the performance, accuracy (mAP), and training efficiency of a real-time detector (YOLO) against a region-proposal based detector (Faster R-CNN).

## 📂 Project Structure
*   `dataset/`: Images and labels in **YOLO format** (normalized `.txt`).
*   `coco_dataset/`: Images and annotations in **COCO format** (`result.json`).
*   `train_yolo.py`: Training script for YOLO11 using the `ultralytics` library.
*   `train_frcnn.py`: Training script for Faster R-CNN using PyTorch/Torchvision.
*   `data.yaml`: Configuration for YOLO dataset paths and classes.

## 🛠 Setup & Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:TheMuffinmuffler/Ais4002-Assignment2.git
   cd Ais4002-Assignment2
   ```
2. Install dependencies (Optimized for NVIDIA GPU):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install ultralytics opencv-python pillow
   ```

## 🏋️ Training
To train the models on a machine with a GPU (e.g., RTX 3070 Ti):

### YOLO11
```bash
python train_yolo.py
```
*Results will be saved in `runs/detect/yolo11_experiment/`.*

### Faster R-CNN
```bash
python train_frcnn.py
```
*Weights will be saved as `faster_rcnn_epoch_X.pth` in the root directory.*

## 📊 Comparison Strategy
- **YOLO11:** Optimized for speed and edge deployment.
- **Faster R-CNN:** Generally more accurate on smaller objects but significantly slower to train and run.

## 📝 Notes
- **Classes:** 8 unique beverage types.
- **Hardware:** Optimized for CUDA-enabled GPUs.
