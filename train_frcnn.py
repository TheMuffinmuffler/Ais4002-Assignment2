import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from PIL import Image

# 1. Custom Dataset class to handle COCO format for Faster R-CNN
class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CocoDataset, self).__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        # CocoDetection returns (image, list of annotations)
        img, target = super(CocoDataset, self).__getitem__(idx)
        
        # Pre-process the target for Faster R-CNN format
        image_id = torch.tensor([idx])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in target:
            # COCO bbox: [xmin, ymin, width, height]
            # Faster R-CNN expects: [xmin, ymin, xmax, ymax]
            x, y, w, h = ann['bbox']
            
            # Filter out invalid boxes (zero area or negative values)
            if w <= 0 or h <= 0:
                continue
                
            boxes.append([x, y, x + w, y + h])
            
            # Faster R-CNN: 0 is background. Labels are 1-based (cat_id + 1).
            labels.append(ann['category_id'] + 1)
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        # If no objects, create a dummy box to avoid crashes (common in training)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }

        if self._transforms is not None:
            img = self._transforms(img)

        return img, target

# 2. Basic Transformations
def get_transform():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

# 3. Training Loop
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    header = f'Epoch: [{epoch}]'
    
    total_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
    print(f"{header} - Average Loss: {total_loss / len(data_loader):.4f}")

def main():
    # Setup Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    # Note: Using your renamed 'coco_dataset' folder
    dataset = CocoDataset(
        root='coco_dataset/images',
        annFile='coco_dataset/result.json',
        transforms=get_transform()
    )
    
    # collate_fn is needed because images have different numbers of objects
    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    # Load pre-trained Faster R-CNN with ResNet-50 FPN backbone
    print("Loading pre-trained Faster R-CNN...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the box predictor for your 8 classes (+ 1 for background)
    num_classes = 9  # 8 drinks + 1 background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)

    # Optimizer & Learning Rate Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()
        
        # Save checkpoint
        torch.save(model.state_dict(), f"faster_rcnn_epoch_{epoch}.pth")
        print(f"Model saved: faster_rcnn_epoch_{epoch}.pth")

    print("Training complete!")

if __name__ == "__main__":
    main()
