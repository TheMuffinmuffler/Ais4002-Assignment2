import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os

if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')

# 1. HARDCODED FRCNN RESULTS (Extracted from validated training logs)
frcnn_v1 = {"mAP50": 0.564, "mAP50-95": 0.457, "time": 4.06}
frcnn_v2 = {"mAP50": 0.957, "mAP50-95": 0.771, "time": 9.56}

def get_best_yolo(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pl.read_csv(csv_path)
    df = df.rename({c: c.strip() for c in df.columns})
    best_row = df.filter(pl.col("metrics/mAP50(B)") == df["metrics/mAP50(B)"].max()).row(0, named=True)
    return {
        "mAP50": best_row['metrics/mAP50(B)'],
        "mAP50-95": best_row['metrics/mAP50-95(B)'],
        "time": df['time'].sum() / 60
    }

yolo_v1 = get_best_yolo(os.path.join("runs", "detect", "yolo_v1_small", "results.csv"))
yolo_v2 = get_best_yolo(os.path.join("runs", "detect", "yolo_v2_large", "results.csv"))

if not yolo_v1 or not yolo_v2:
    print("Warning: YOLO results not found. Using fallback hardcoded values for demonstration.")
    yolo_v1 = {"mAP50": 0.962, "mAP50-95": 0.795, "time": 3.94}
    yolo_v2 = {"mAP50": 0.969, "mAP50-95": 0.818, "time": 6.61}

# Styling
yolo_blue = '#0477BF'
frcnn_red = '#D93D4A'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.alpha'] = 0.3

def create_bar_plot(yolo_data, frcnn_data, current_labels, ylabel, title, filename, ylim=(0, 1.1)):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(current_labels))
    width = 0.35
    rects1 = ax.bar(x - width/2, yolo_data, width, label='YOLOv11s', color=yolo_blue, edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, frcnn_data, width, label='Faster R-CNN', color=frcnn_red, edgecolor='black', linewidth=0.5)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(current_labels, fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc='upper left')
    ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}' if h < 1 else f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    fig.tight_layout()
    os.makedirs('final_report_assets', exist_ok=True)
    plt.savefig(os.path.join('final_report_assets', filename), bbox_inches='tight')
    plt.close()

labels = ['Dataset V1 (Baseline)', 'Dataset V2 (Scaled)']
yolo_m50 = [yolo_v1['mAP50'], yolo_v2['mAP50']]
frcnn_m50 = [frcnn_v1['mAP50'], frcnn_v2['mAP50']]
yolo_m95 = [yolo_v1['mAP50-95'], yolo_v2['mAP50-95']]
frcnn_m95 = [frcnn_v1['mAP50-95'], frcnn_v2['mAP50-95']]
yolo_t = [yolo_v1['time'], yolo_v2['time']]
frcnn_t = [frcnn_v1['time'], frcnn_v2['time']]

create_bar_plot(yolo_m50, frcnn_m50, labels, 'mAP @ 0.50', 'Accuracy Comparison (mAP50)', 'comparison_accuracy_final.png')
create_bar_plot(yolo_m95, frcnn_m95, labels, 'mAP @ 0.50:0.95', 'Precision Comparison (mAP50-95)', 'comparison_precision_final.png')
create_bar_plot(yolo_t, frcnn_t, labels, 'Minutes', 'Training Efficiency Comparison', 'comparison_time_final.png', ylim=(0, 12))
