import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from PIL import Image

# 1. CUSTOM DATASET CLASS: Handles COCO-formatted annotations for Faster R-CNN
class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        # Initialize the COCO dataset by calling the parent class constructor
        super(CocoDataset, self).__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        # Retrieve the image and raw COCO target (list of dictionaries)
        img, target = super(CocoDataset, self).__getitem__(idx)
        
        image_id = torch.tensor([idx]) # Every image must have a unique ID
        boxes = [] # Initialize list for bounding box coordinates
        labels = [] # Initialize list for class labels
        areas = [] # Initialize list for object areas
        iscrowd = [] # Initialize list for crowd information (for COCO standard)
        
        for ann in target:
            # COCO bbox format is [xmin, ymin, width, height]
            # Faster R-CNN requires absolute coordinates: [xmin, ymin, xmax, ymax]
            x, y, w, h = ann['bbox']
            
            # Skip invalid or corrupted annotations (zero width or height)
            if w <= 0 or h <= 0:
                continue
                
            # Convert to absolute [xmin, ymin, xmax, ymax] format
            boxes.append([x, y, x + w, y + h])
            
            # Label 0 is reserved for background in Faster R-CNN
            # So we shift all IDs up by 1 (cat_id 0 becomes label 1)
            labels.append(ann['category_id'] + 1)
            areas.append(ann['area']) # Useful for performance metrics
            iscrowd.append(ann['iscrowd']) # Marks if objects are overlapping in a crowd

        # Convert the Python lists into PyTorch tensors (the format the GPU uses)
        if len(boxes) == 0: # Handle cases with no objects to avoid training errors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Assemble the dictionary in the exact format required by Torchvision's Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }

        # Apply image transformations (normalization, scaling)
        if self._transforms is not None:
            img = self._transforms(img)

        return img, target

# 2. IMAGE TRANSFORMS: Prepares raw images for the neural network
def get_transform():
    return T.Compose([
        T.ToImage(), # Convert PIL image to PyTorch Image object
        T.ToDtype(torch.float32, scale=True), # Normalize pixel values to 0.0 - 1.0
    ])

# 3. EPOCH TRAINING: Processes one full pass of the data
def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train() # Put the model in training mode (enables gradient tracking)
    total_loss = 0
    
    # Process the images in batches
    for images, targets in data_loader:
        # Transfer both images and labels to the GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Use Automatic Mixed Precision (AMP) for faster training
        with torch.amp.autocast(device_type=device.type):
            # Calculate the four types of loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # BACKPROPAGATION with AMP Scaler
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += losses.item()
        
    print(f"Epoch: [{epoch}] - Average Loss: {total_loss / len(data_loader):.4f}")

import argparse
import os
import torch
...
def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
...
    print(f"Epoch: [{epoch}] - Average Loss: {total_loss / len(data_loader):.4f}")

def train_frcnn(dataset_root='coco_dataset', ann_file='coco_dataset/result.json', experiment_name='frcnn_experiment'):
    # Detect if NVIDIA CUDA is available (Optimized for your 3070 Ti)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = os.path.join('runs', 'frcnn', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Scaler for AMP
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Initialize the dataset using the COCO-formatted folder we set up
    dataset = CocoDataset(
        root=dataset_root,
        annFile=ann_file,
        transforms=get_transform()
    )
    
    # Custom collate function needed because each image has a different number of boxes
    def collate_fn(batch):
        return tuple(zip(*batch))

    # DataLoader handles shuffling the data and feeding it in small batches
    data_loader = DataLoader(
        dataset, 
        batch_size=4, # Set to 4 to save memory; can be increased to 8 or 12 on 3070 Ti
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # LOAD MODEL: Start with a ResNet-50 Faster R-CNN model pretrained on COCO
    print("Loading pre-trained Faster R-CNN ResNet-50...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the output layer to match your 8 specific beverage classes
    # 9 = 8 Beverages + 1 Background (always needed)
    num_classes = 9 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device) # Move the entire model to the GPU

    # OPTIMIZER: Stochastic Gradient Descent with momentum and weight decay
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # LEARNING RATE SCHEDULER: Slowly reduces the learning rate to fine-tune results
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # MAIN TRAINING LOOP: 10 Epochs is a good starting point for transfer learning
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, scaler)
        lr_scheduler.step() # Update the learning rate
        
        # Save the model state as a checkpoint
        checkpoint_path = os.path.join(output_dir, f"faster_rcnn_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(output_dir, "best.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    parser.add_argument("--root", type=str, default="coco_dataset", help="Dataset root directory")
    parser.add_argument("--ann", type=str, default="coco_dataset/result.json", help="Annotation file path")
    parser.add_argument("--name", type=str, default="frcnn_experiment", help="Experiment name")
    args = parser.parse_args()
    
    train_frcnn(dataset_root=args.root, ann_file=args.ann, experiment_name=args.name)
