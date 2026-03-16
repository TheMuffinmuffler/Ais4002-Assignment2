import matplotlib.pyplot as plt
import numpy as np
import os

# Data Definition
labels = ['Dataset V1 (Baseline)', 'Dataset V2 (Scaled)']
yolo_mAP50 = [0.962, 0.969]
frcnn_mAP50 = [0.564, 0.957]

yolo_mAP50_95 = [0.795, 0.818]
frcnn_mAP50_95 = [0.457, 0.771]

# YOLO times: V1 = 236.6s (~3.94m), V2 = 396.8s (~6.61m)
yolo_time = [3.94, 6.61]
frcnn_time = [4.06, 9.56]

# Styling Configuration (YOLO Aesthetic)
yolo_blue = '#0477BF' # Custom blue
frcnn_red = '#D93D4A'  # Custom red
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#f8f9fa' # Soft light grey
plt.rcParams['grid.alpha'] = 0.3

def create_bar_plot(yolo_data, frcnn_data, current_labels, ylabel, title, filename, ylim=(0, 1.1)):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(current_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, yolo_data, width, label='YOLOv11s', color=yolo_blue, edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, frcnn_data, width, label='Faster R-CNN', color=frcnn_red, edgecolor='black', linewidth=0.5)

    # Styling
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(current_labels, fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc='upper left')
    ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Value Labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Clean Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    os.makedirs('final_report_assets', exist_ok=True)
    save_path = os.path.join('final_report_assets', filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close()

# 1. Combined Plots
create_bar_plot(yolo_mAP50, frcnn_mAP50, labels, 'mAP @ 0.50', 'Accuracy Comparison (mAP50)', 'comparison_accuracy_combined.png')
create_bar_plot(yolo_mAP50_95, frcnn_mAP50_95, labels, 'mAP @ 0.50:0.95', 'Precision Comparison (mAP50-95)', 'comparison_precision_combined.png')
create_bar_plot(yolo_time, frcnn_time, labels, 'Minutes', 'Training Efficiency Comparison', 'comparison_time_combined.png', ylim=(0, 12))

# 2. Dataset V1 Only (Metrics Comparison)
create_bar_plot([yolo_mAP50[0], yolo_mAP50_95[0]], [frcnn_mAP50[0], frcnn_mAP50_95[0]], 
                ['mAP50', 'mAP50-95'], 'Score', 'Dataset V1: Performance Summary', 'comparison_v1_metrics.png')

# 3. Dataset V2 Only (Metrics Comparison)
create_bar_plot([yolo_mAP50[1], yolo_mAP50_95[1]], [frcnn_mAP50[1], frcnn_mAP50_95[1]], 
                ['mAP50', 'mAP50-95'], 'Score', 'Dataset V2: Performance Summary', 'comparison_v2_metrics.png')
