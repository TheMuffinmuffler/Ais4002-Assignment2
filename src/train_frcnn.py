import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 1. CUSTOM DATASET CLASS
class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CocoDataset, self).__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDataset, self).__getitem__(idx)
        # Use the actual COCO image ID instead of the index
        image_id = torch.tensor([self.ids[idx]])
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

@torch.no_grad()
def evaluate(model, data_loader, device, ann_file):
    model.eval()
    coco_gt = COCO(ann_file)
    results = []
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        
        for i, output in enumerate(outputs):
            image_id = targets[i]['image_id'].item()
            # Convert internal labels back to original category IDs (subtract 1)
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy() - 1
            
            for box, score, label in zip(boxes, scores, labels):
                # Faster R-CNN uses [x1, y1, x2, y2], COCO uses [x, y, w, h]
                x1, y1, x2, y2 = box
                results.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(score)
                })
    
    if not results:
        print("No detections found for evaluation.")
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Return mAP @ IoU=0.50:0.95 (first element in stats)
    return coco_eval.stats[0]

def train_frcnn(dataset_root='coco_dataset', train_ann='coco_dataset/train.json', val_ann='coco_dataset/val.json', experiment_name='frcnn_experiment'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    output_dir = os.path.join('runs', 'frcnn', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    def collate_fn(batch): return tuple(zip(*batch))
    
    train_dataset = CocoDataset(root=dataset_root, annFile=train_ann, transforms=get_transform())
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    val_dataset = CocoDataset(root=dataset_root, annFile=val_ann, transforms=get_transform())
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 9 # 8 types + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    num_epochs = 25
    loss_history = []
    map_history = []
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        loss_history.append(avg_loss)
        lr_scheduler.step()
        
        print(f"Validating Epoch {epoch}...")
        mAP = evaluate(model, val_loader, device, val_ann)
        map_history.append(mAP)

        # We only save the final model at the end to keep project size small

    # Plot results
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(num_epochs), loss_history, marker='o', color='tab:red', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('mAP (0.5:0.95)', color='tab:blue')
    ax2.plot(range(num_epochs), map_history, marker='s', color='tab:blue', label='mAP')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'Faster R-CNN Training - {experiment_name}')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_curve.png'))
    plt.close()

    final_path = os.path.join(output_dir, "best.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Results saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    parser.add_argument("--root", type=str, default="coco_dataset", help="Dataset root directory")
    parser.add_argument("--train_ann", type=str, default="coco_dataset/train.json", help="Train annotation file path")
    parser.add_argument("--val_ann", type=str, default="coco_dataset/val.json", help="Val annotation file path")
    parser.add_argument("--name", type=str, default="frcnn_experiment", help="Experiment name")
    args = parser.parse_args()
    
    train_frcnn(dataset_root=args.root, train_ann=args.train_ann, val_ann=args.val_ann, experiment_name=args.name)

