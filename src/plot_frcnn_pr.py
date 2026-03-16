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

# Fix working directory
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
sys.path.append(os.getcwd())
from src.train_frcnn import CocoDataset, get_transform

def get_pr_data_per_class(model_path, dataset_root, ann_file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    backbone = resnet_fpn_backbone('resnet50', weights=None, trainable_layers=0)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=9)
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return None, None
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

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

    print(f"Running inference to collect PR curve data for {ann_file}...")
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                gt_ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
                gt_anns = coco_gt.loadAnns(gt_ann_ids)
                
                dt_boxes = output['boxes'].cpu().numpy()
                dt_scores = output['scores'].cpu().numpy()
                dt_labels = output['labels'].cpu().numpy() - 1 
                
                matched_gt = set()
                img_detections = sorted(zip(dt_boxes, dt_scores, dt_labels), key=lambda x: x[1], reverse=True)
                
                for box, score, label in img_detections:
                    best_iou, best_gt_idx = 0, -1
                    for idx, gt in enumerate(gt_anns):
                        if gt['category_id'] != label or idx in matched_gt: continue
                        gx, gy, gw, gh = gt['bbox']
                        x1, y1, x2, y2 = box
                        inter_x1, inter_y1 = max(x1, gx), max(y1, gy)
                        inter_x2, inter_y2 = min(x2, gx + gw), min(y2, gy + gh)
                        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        iou = inter_area / ((x2-x1)*(y2-y1) + gw*gh - inter_area + 1e-6)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                    
                    if best_iou >= 0.5:
                        detections_per_class[label].append((score, 1))
                        matched_gt.add(best_gt_idx)
                    else:
                        detections_per_class[label].append((score, 0))

    pr_curves = {}
    mean_ap = 0
    valid_classes = 0

    for cat_id in cat_ids:
        dts = sorted(detections_per_class[cat_id], key=lambda x: x[0], reverse=True)
        total_gt = gt_counts_per_class[cat_id]
        
        if total_gt == 0:
            continue
            
        tps = np.array([x[1] for x in dts])
        fps = 1 - tps
        
        acc_tps = np.cumsum(tps)
        acc_fps = np.cumsum(fps)
        
        recalls = acc_tps / total_gt
        precisions = np.divide(acc_tps, (acc_tps + acc_fps), out=np.zeros_like(acc_tps, dtype=float), where=(acc_tps + acc_fps) != 0)
        
        # Smooth precision curve (make it monotonically decreasing)
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
            
        # Add a starting point at recall 0 to make the graph look complete
        recalls = np.concatenate(([0.0], recalls))
        precisions = np.concatenate(([precisions[0] if len(precisions)>0 else 1.0], precisions))

        # Calculate AP (Area Under Curve) using COCO 101-point interpolation
        recall_levels = np.linspace(0.0, 1.00, 101)
        ap = 0.0
        for r in recall_levels:
            idx = np.where(recalls >= r)[0]
            if len(idx) > 0:
                ap += np.max(precisions[idx])
        ap /= 101.0
        
        pr_curves[cat_names[cat_id]] = {'recall': recalls, 'precision': precisions, 'ap': ap}
        mean_ap += ap
        valid_classes += 1
        
    mAP = mean_ap / valid_classes if valid_classes > 0 else 0
    
    return pr_curves, mAP

def plot_pr_for_dataset(model_path, dataset_root, ann_file, title, filename):
    pr_curves, mAP = get_pr_data_per_class(model_path, dataset_root, ann_file)
    if pr_curves is None: return
    
    plt.figure(figsize=(10, 7), dpi=150)
    
    # Plot individual class lines
    for name, data in pr_curves.items():
        plt.plot(data['recall'], data['precision'], label=f"{name} {data['ap']:.3f}", linewidth=1.5, alpha=0.8)
    
    # Plot a bold mean AP line (average of precisions at each recall point)
    # We interpolate to a common recall grid to average
    common_recall = np.linspace(0, 1, 101)
    mean_precision = np.zeros(101)
    for name, data in pr_curves.items():
        interp_p = np.interp(common_recall, data['recall'], data['precision'])
        mean_precision += interp_p
    mean_precision /= len(pr_curves)
    
    plt.plot(common_recall, mean_precision, color='black', linewidth=3, label=f"all classes mAP@0.5 {mAP:.3f}")
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    
    # Legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    os.makedirs('final_report_assets', exist_ok=True)
    plt.savefig(os.path.join('final_report_assets', filename), bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to 'final_report_assets/{filename}'")

if __name__ == "__main__":
    # Dataset V1
    plot_pr_for_dataset(
        model_path="runs/frcnn/frcnn_v1_validated/best.pth",
        dataset_root="data/coco_dataset_v1",
        ann_file="data/coco_dataset_v1/val.json",
        title='Faster R-CNN: Precision-Recall Curve (V1 Baseline)',
        filename='frcnn_PR_curve_v1.png'
    )
    # Dataset V2
    plot_pr_for_dataset(
        model_path="runs/frcnn/frcnn_v2_validated/best.pth",
        dataset_root="data/coco_dataset_v2",
        ann_file="data/coco_dataset_v2/val.json",
        title='Faster R-CNN: Precision-Recall Curve (V2 Scaled)',
        filename='frcnn_PR_curve_v2.png'
    )
