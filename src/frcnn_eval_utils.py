import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import numpy as np
import os
from collections import defaultdict
from src.train_frcnn import CocoDataset, get_transform

def load_frcnn_model(model_path, device):
    # Load model
    backbone = resnet_fpn_backbone('resnet50', weights=None, trainable_layers=0)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=9)
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return None
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def get_frcnn_detections(model, data_loader, device, coco_gt):
    detections_per_class = defaultdict(list)
    gt_counts_per_class = defaultdict(int)
    cat_ids = coco_gt.getCatIds()
    
    for cat_id in cat_ids:
        gt_counts_per_class[cat_id] = len(coco_gt.getAnnIds(catIds=[cat_id]))

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                gt_ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
                gt_anns = coco_gt.loadAnns(gt_ann_ids)
                dt_boxes, dt_scores, dt_labels = output['boxes'].cpu().numpy(), output['scores'].cpu().numpy(), output['labels'].cpu().numpy() - 1
                matched_gt = set()
                img_detections = sorted(zip(dt_boxes, dt_scores, dt_labels), key=lambda x: x[1], reverse=True)
                for box, score, label in img_detections:
                    best_iou, best_gt_idx = 0, -1
                    for idx, gt in enumerate(gt_anns):
                        if gt['category_id'] != label or idx in matched_gt: continue
                        gx, gy, gw, gh = gt['bbox']
                        x1, y1, x2, y2 = box
                        inter_x1, inter_y1, inter_x2, inter_y2 = max(x1, gx), max(y1, gy), min(x2, gx + gw), min(y2, gy + gh)
                        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        iou = inter_area / ((x2-x1)*(y2-y1) + gw*gh - inter_area + 1e-6)
                        if iou > best_iou: best_iou, best_gt_idx = iou, idx
                    if best_iou >= 0.5:
                        detections_per_class[label].append((score, 1))
                        matched_gt.add(best_gt_idx)
                    else:
                        detections_per_class[label].append((score, 0))
    return detections_per_class, gt_counts_per_class

def get_cocotools_results(model, data_loader, device):
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy() - 1
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        'score': float(score)
                    })
    return results
