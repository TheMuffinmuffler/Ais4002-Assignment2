import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from PIL import Image
import argparse
import matplotlib.pyplot as plt

# 1. CUSTOM DATASET CLASS
class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CocoDataset, self).__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDataset, self).__getitem__(idx)
        image_id = torch.tensor([idx])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in target:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0: continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] + 1)
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

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

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': areas, 'iscrowd': iscrowd}
        if self._transforms is not None:
            img = self._transforms(img)
        return img, target

def get_transform():
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast(device_type=device.type):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += losses.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch: [{epoch}] - Average Loss: {avg_loss:.4f}")
    return avg_loss

def train_frcnn(dataset_root='coco_dataset', ann_file='coco_dataset/result.json', experiment_name='frcnn_experiment'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    output_dir = os.path.join('runs', 'frcnn', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    dataset = CocoDataset(root=dataset_root, annFile=ann_file, transforms=get_transform())
    def collate_fn(batch): return tuple(zip(*batch))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 9 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    loss_history = []
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, scaler)
        loss_history.append(avg_loss)
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch}.pth"))

    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), loss_history, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Faster R-CNN Training Loss - {experiment_name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    final_path = os.path.join(output_dir, "best.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Results saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    parser.add_argument("--root", type=str, default="coco_dataset", help="Dataset root directory")
    parser.add_argument("--ann", type=str, default="coco_dataset/result.json", help="Annotation file path")
    parser.add_argument("--name", type=str, default="frcnn_experiment", help="Experiment name")
    args = parser.parse_args()
    train_frcnn(dataset_root=args.root, ann_file=args.ann, experiment_name=args.name)
