from ultralytics import YOLO  # Import the Ultralytics YOLO library
import torch  # Import PyTorch for device management

def train_yolo():
    # Detect the best available hardware for training
    # 'cuda' for your RTX 3070 Ti, 'mps' for Mac Silicon, or 'cpu' as a fallback
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the YOLO11 model using pretrained 'nano' weights
    # Pretrained weights provide a 'head start' so the model learns faster
    model = YOLO("yolo11n.pt")

    # Start the training process with the following configuration:
    # data: Points to your data.yaml which defines image paths and class names
    # epochs: The number of full passes through the entire dataset (100 is standard)
    # imgsz: Scales all images to 640x640 pixels during training
    # batch: Number of images processed at once (16 is optimized for 8GB VRAM)
    # device: Forces the model to use the GPU/CPU detected earlier
    # name: The folder name where your results and weights will be saved
    results = model.train(
        data="data.yaml", 
        epochs=100, 
        imgsz=640, 
        batch=16,
        device=device,
        name="yolo11_experiment"
    )

    # Inform the user where the final weights (best.pt) are located
    print("Training complete! Your model is saved in the 'runs/detect/yolo11_experiment/weights/best.pt' folder.")

if __name__ == "__main__":
    train_yolo()  # Execute the training function
