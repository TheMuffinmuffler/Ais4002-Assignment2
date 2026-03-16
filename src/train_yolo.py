import argparse
from ultralytics import YOLO
import torch

def train_yolo(data_yaml="data.yaml", experiment_name="yolo11_experiment"):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolo11s.pt")

    results = model.train(
        data=data_yaml, 
        epochs=100, 
        imgsz=640, 
        batch=16,
        device=device,
        name=experiment_name
    )

    print(f"Training complete! Your model is saved in the 'runs/detect/{experiment_name}/weights/best.pt' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--name", type=str, default="yolo11_experiment", help="Experiment name")
    args = parser.parse_args()
    
    train_yolo(data_yaml=args.data, experiment_name=args.name)
