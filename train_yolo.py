from ultralytics import YOLO
import torch

def train_yolo():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a pretrained YOLO11 model (nano version for speed)
    # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    model = YOLO("yolo11n.pt")

    # Train the model
    # data: path to your data.yaml
    # epochs: how many times to see the entire dataset
    # imgsz: input image size (standard is 640)
    # batch: number of images per batch (increase for your 3070ti)
    results = model.train(
        data="data.yaml", 
        epochs=100, 
        imgsz=640, 
        batch=16, 
        device=device,
        name="yolo11_experiment"
    )

    print("Training complete! Your model is saved in the 'runs/detect/train' folder.")

if __name__ == "__main__":
    train_yolo()
