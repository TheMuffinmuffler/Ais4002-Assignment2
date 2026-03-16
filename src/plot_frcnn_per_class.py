import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import os

import sys

# Fix working directory if run from src/
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')

# Add src to path so we can import from it
sys.path.append(os.getcwd())
from src.train_frcnn import CocoDataset, get_transform

def evaluate_per_class(model_path, dataset_root, ann_file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load model
    # We use weights_backbone=None to avoid downloading from the internet
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    backbone = resnet_fpn_backbone('resnet50', weights=None, trainable_layers=0)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=9)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load data
    dataset = CocoDataset(root=dataset_root, annFile=ann_file, transforms=get_transform())
    def collate_fn(batch): return tuple(zip(*batch))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    coco_gt = COCO(ann_file)
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

    coco_dt = coco_gt.loadRes(results)
    
    # Get Class Names
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    cat_ids = [cat['id'] for cat in cats]
    
    per_class_map = []
    
    print("\nCalculating per-class mAP50...")
    for cat_id in cat_ids:
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # coco_eval.stats[1] is mAP @ IoU=0.50
        per_class_map.append(coco_eval.stats[1])

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.barh(cat_names, per_class_map, color='tab:red')
    plt.xlabel('mAP @ 0.50')
    plt.title('Faster R-CNN: Per-Class Accuracy (V2 Scaled)')
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(per_class_map):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.tight_layout()
    os.makedirs('final_report_assets', exist_ok=True)
    plt.savefig('final_report_assets/frcnn_per_class.png')
    print("Per-class plot saved to 'final_report_assets/frcnn_per_class.png'")

if __name__ == "__main__":
    evaluate_per_class(
        model_path="runs/frcnn/frcnn_v2_validated/best.pth",
        dataset_root="data/coco_dataset_v2",
        ann_file="data/coco_dataset_v2/val.json"
    )
