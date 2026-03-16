import argparse
from ultralytics import YOLO  # Import the Ultralytics YOLO library
import torch  # Import PyTorch for device management

def train_yolo(data_yaml="data.yaml", experiment_name="yolo11_experiment"):
    # Detect the best available hardware for training
    # 'cuda' for your RTX 3070 Ti, 'mps' for Mac Silicon, or 'cpu' as a fallback
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the YOLO11 model using pretrained 'small' weights
    # Pretrained weights provide a 'head start' so the model learns faster
    model = YOLO("yolo11s.pt")

    # Start the training process with the following configuration:
    # data: Points to your data.yaml which defines image paths and class names
    # epochs: The number of full passes through the entire dataset (100 is standard)
    # imgsz: Scales all images to 640x640 pixels during training
    # batch: Number of images processed at once (16 is optimized for 8GB VRAM)
    # device: Forces the model to use the GPU/CPU detected earlier
    # name: The folder name where your results and weights will be saved
    results = model.train(
        data=data_yaml, 
        epochs=100, 
        imgsz=640, 
        batch=16,
        device=device,
        name=experiment_name
    )

    # Inform the user where the final weights (best.pt) are located
    print(f"Training complete! Your model is saved in the 'runs/detect/{experiment_name}/weights/best.pt' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--name", type=str, default="yolo11_experiment", help="Experiment name")
    args = parser.parse_args()
    
    train_yolo(data_yaml=args.data, experiment_name=args.name)  # Execute the training function
