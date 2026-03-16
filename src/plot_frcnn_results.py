import torch
import torchvision
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
sys.path.append(os.getcwd())

from src.train_frcnn import CocoDataset, get_transform
from src.frcnn_eval_utils import load_frcnn_model, get_frcnn_detections, get_cocotools_results

def plot_p_r_curves(conf, all_metrics, suffix):
    for metric_key, ylabel, title_prefix, filename_suffix in [
        ("P", "Precision", "Precision-Confidence Curve", "P_curve"),
        ("R", "Recall", "Recall-Confidence Curve", "R_curve")
    ]:
        plt.figure(figsize=(10, 7), dpi=150)
        metric_matrix = np.array([m[metric_key] for m in all_metrics.values()])
        mean_metric = np.mean(metric_matrix, axis=0)
        
        for name, m in all_metrics.items():
            plt.plot(conf, m[metric_key], label=f'{name} {m[metric_key][0]:.2f}', linewidth=1, alpha=0.5)
            
        plt.plot(conf, mean_metric, color='black', linewidth=3, label=f'all classes {mean_metric[0]:.2f}')
        plt.title(f'Faster R-CNN: {title_prefix} ({suffix.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence'), plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(0, 1), plt.ylim(0, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.savefig(f'final_report_assets/frcnn_{filename_suffix}_{suffix}.png', bbox_inches='tight')
        plt.close()

def plot_f1_curve(conf, all_class_f1s, suffix):
    plt.figure(figsize=(10, 7), dpi=150)
    f1_matrix = np.array(list(all_class_f1s.values()))
    mean_f1 = np.mean(f1_matrix, axis=0)
    
    for name, f1_curve in all_class_f1s.items():
        plt.plot(conf, f1_curve, label=f'{name} {max(f1_curve):.2f}', linewidth=1, alpha=0.5)
        
    best_mean_f1, best_conf = max(mean_f1), conf[np.argmax(mean_f1)]
    plt.plot(conf, mean_f1, color='black', linewidth=3, label=f'all classes {best_mean_f1:.2f} at {best_conf:.2f}')
    plt.plot(best_conf, best_mean_f1, 'ro', markersize=8)
    plt.title(f'Faster R-CNN: F1-Confidence Curve ({suffix.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence'), plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1), plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.savefig(f'final_report_assets/frcnn_f1_curve_{suffix}.png', bbox_inches='tight')
    plt.close()

def plot_pr_curve(pr_curves, mAP, suffix):
    plt.figure(figsize=(10, 7), dpi=150)
    for name, data in pr_curves.items():
        plt.plot(data['recall'], data['precision'], label=f"{name} {data['ap']:.3f}", linewidth=1.5, alpha=0.8)
    
    common_recall = np.linspace(0, 1, 101)
    mean_precision = np.zeros(101)
    for name, data in pr_curves.items():
        interp_p = np.interp(common_recall, data['recall'], data['precision'])
        mean_precision += interp_p
    mean_precision /= len(pr_curves)
    
    plt.plot(common_recall, mean_precision, color='black', linewidth=3, label=f"all classes mAP@0.5 {mAP:.3f}")
    plt.title(f'Faster R-CNN: Precision-Recall Curve ({suffix.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Recall'), plt.ylabel('Precision')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1.0), plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.savefig(f'final_report_assets/frcnn_PR_curve_{suffix}.png', bbox_inches='tight')
    plt.close()

def plot_per_class_map(cat_names, per_class_map, suffix):
    plt.figure(figsize=(12, 7))
    plt.barh(cat_names, per_class_map, color='tab:red')
    plt.xlabel('mAP @ 0.50')
    plt.title(f'Faster R-CNN: Per-Class Accuracy ({suffix.upper()})')
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    for i, v in enumerate(per_class_map):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    plt.tight_layout()
    plt.savefig(f'final_report_assets/frcnn_per_class_{suffix}.png')
    plt.close()

def main():
    os.makedirs('final_report_assets', exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    configs = [
        ("runs/frcnn/frcnn_v1_validated/best.pth", "data/coco_dataset_v1", "v1"),
        ("runs/frcnn/frcnn_v2_validated/best.pth", "data/coco_dataset_v2", "v2")
    ]
    
    for model_path, data_root, suffix in configs:
        print(f"\nProcessing {suffix.upper()}...")
        ann_file = os.path.join(data_root, "val.json")
        coco_gt = COCO(ann_file)
        cat_ids = coco_gt.getCatIds()
        cat_names_dict = {cat['id']: cat['name'] for cat in coco_gt.loadCats(cat_ids)}
        
        model = load_frcnn_model(model_path, device)
        if model is None: continue
        
        dataset = CocoDataset(root=data_root, annFile=ann_file, transforms=get_transform())
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        
        # 1. PR/F1/P/R Curves data
        detections_per_class, gt_counts_per_class = get_frcnn_detections(model, data_loader, device, coco_gt)
        
        conf_levels = np.linspace(0, 1, 100)
        p_r_metrics = {}
        f1_metrics = {}
        pr_curves_data = {}
        total_ap = 0
        
        for cat_id in cat_ids:
            dts = sorted(detections_per_class[cat_id], key=lambda x: x[0], reverse=True)
            total_gt = gt_counts_per_class[cat_id]
            
            p_scores, r_scores, f1_scores = [], [], []
            for conf in conf_levels:
                tp = sum(1 for s, is_tp in dts if s >= conf and is_tp == 1)
                fp = sum(1 for s, is_tp in dts if s >= conf and is_tp == 0)
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (total_gt + 1e-6)
                p_scores.append(precision)
                r_scores.append(recall)
                f1_scores.append(2 * (precision * recall) / (precision + recall + 1e-6))
            
            p_r_metrics[cat_names_dict[cat_id]] = {"P": p_scores, "R": r_scores}
            f1_metrics[cat_names_dict[cat_id]] = f1_scores
            
            # PR Curve specific (VOC style)
            if total_gt > 0:
                tps = np.array([x[1] for x in dts])
                fps = 1 - tps
                acc_tps, acc_fps = np.cumsum(tps), np.cumsum(fps)
                recalls = acc_tps / total_gt
                precisions = np.divide(acc_tps, (acc_tps + acc_fps), out=np.zeros_like(acc_tps, dtype=float), where=(acc_tps + acc_fps) != 0)
                for i in range(len(precisions) - 1, 0, -1): precisions[i - 1] = max(precisions[i - 1], precisions[i])
                recalls = np.concatenate(([0.0], recalls))
                precisions = np.concatenate(([precisions[0] if len(precisions)>0 else 1.0], precisions))
                
                ap = 0.0
                for r in np.linspace(0.0, 1.00, 101):
                    idx = np.where(recalls >= r)[0]
                    if len(idx) > 0: ap += np.max(precisions[idx])
                ap /= 101.0
                pr_curves_data[cat_names_dict[cat_id]] = {'recall': recalls, 'precision': precisions, 'ap': ap}
                total_ap += ap

        mAP_05 = total_ap / len(cat_ids)
        
        # Plotting
        plot_p_r_curves(conf_levels, p_r_metrics, suffix)
        plot_f1_curve(conf_levels, f1_metrics, suffix)
        plot_pr_curve(pr_curves_data, mAP_05, suffix)
        
        # 2. Per-class mAP using COCOeval
        results = get_cocotools_results(model, data_loader, device)
        if results:
            coco_dt = coco_gt.loadRes(results)
            per_class_map = []
            for cat_id in cat_ids:
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.params.catIds = [cat_id]
                coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
                per_class_map.append(coco_eval.stats[1])
            plot_per_class_map([cat_names_dict[cid] for cid in cat_ids], per_class_map, suffix)

if __name__ == "__main__":
    main()
