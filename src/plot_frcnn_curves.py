import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import defaultdict

# Fix working directory if run from src/
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
sys.path.append(os.getcwd())
from src.train_frcnn import CocoDataset, get_transform

def get_metrics_data(model_path, dataset_root, ann_file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load model
    backbone = resnet_fpn_backbone('resnet50', weights=None, trainable_layers=0)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=9)
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return None, None
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load data
    dataset = CocoDataset(root=dataset_root, annFile=ann_file, transforms=get_transform())
    def collate_fn(batch): return tuple(zip(*batch))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    coco_gt = COCO(ann_file)
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    cat_names = {cat['id']: cat['name'] for cat in cats}
    
    detections_per_class = defaultdict(list)
    gt_counts_per_class = defaultdict(int)
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

    conf_levels = np.linspace(0, 1, 100)
    all_class_metrics = {}
    for cat_id in cat_ids:
        dts = sorted(detections_per_class[cat_id], key=lambda x: x[0], reverse=True)
        total_gt = gt_counts_per_class[cat_id]
        p_scores = []
        r_scores = []
        for conf in conf_levels:
            tp = sum(1 for s, is_tp in dts if s >= conf and is_tp == 1)
            fp = sum(1 for s, is_tp in dts if s >= conf and is_tp == 0)
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (total_gt + 1e-6)
            p_scores.append(precision)
            r_scores.append(recall)
        all_class_metrics[cat_names[cat_id]] = {"P": p_scores, "R": r_scores}
    return conf_levels, all_class_metrics

def plot_curve(conf, all_metrics, metric_key, title, filename):
    plt.figure(figsize=(10, 7), dpi=150)
    metric_matrix = np.array([m[metric_key] for m in all_metrics.values()])
    mean_metric = np.mean(metric_matrix, axis=0)
    
    for name, m in all_metrics.items():
        plt.plot(conf, m[metric_key], label=f'{name} {m[metric_key][0]:.2f}', linewidth=1, alpha=0.5)
        
    plt.plot(conf, mean_metric, color='black', linewidth=3, label=f'all classes {mean_metric[0]:.2f}')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Confidence'), plt.ylabel('Precision' if metric_key == "P" else "Recall")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1), plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    os.makedirs('final_report_assets', exist_ok=True)
    plt.savefig(os.path.join('final_report_assets', filename), bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {filename}")

if __name__ == "__main__":
    configs = [
        ("runs/frcnn/frcnn_v1_validated/best.pth", "data/coco_dataset_v1", "v1"),
        ("runs/frcnn/frcnn_v2_validated/best.pth", "data/coco_dataset_v2", "v2")
    ]
    
    for model_path, data_root, suffix in configs:
        ann_file = os.path.join(data_root, "val.json")
        conf, metrics = get_metrics_data(model_path, data_root, ann_file)
        if conf is not None:
            plot_curve(conf, metrics, "P", f'Faster R-CNN: Precision-Confidence Curve ({suffix.upper()})', f'frcnn_P_curve_{suffix}.png')
            plot_curve(conf, metrics, "R", f'Faster R-CNN: Recall-Confidence Curve ({suffix.upper()})', f'frcnn_R_curve_{suffix}.png')
