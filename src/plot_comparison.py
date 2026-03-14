import polars as pl
import matplotlib.pyplot as plt
import os

# Fix working directory if run from src/
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')

# 1. HARDCODED FRCNN RESULTS (Extracted from the validated training logs)
# Dataset V1
frcnn_v1 = {
    "name": "FRCNN V1 (Baseline)",
    "mAP50": 0.564,
    "mAP50-95": 0.457,
    "training_time": 4.06,  # minutes
}

# Dataset V2
frcnn_v2 = {
    "name": "FRCNN V2 (Scaled)",
    "mAP50": 0.957,
    "mAP50-95": 0.771,
    "training_time": 9.56,  # minutes
}

# 2. READ YOLO RESULTS FROM CSV
def get_best_yolo(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pl.read_csv(csv_path)
    # Strip column names
    df = df.rename({c: c.strip() for c in df.columns})
    
    # Find row with max mAP50
    best_row = df.filter(pl.col("metrics/mAP50(B)") == df["metrics/mAP50(B)"].max()).row(0, named=True)
    
    return {
        "mAP50": best_row['metrics/mAP50(B)'],
        "mAP50-95": best_row['metrics/mAP50-95(B)'],
        "training_time": df['time'].sum() / 60  # total time in minutes
    }

yolo_v1 = get_best_yolo(os.path.join("runs", "detect", "yolo_v1_small", "results.csv"))
yolo_v2 = get_best_yolo(os.path.join("runs", "detect", "yolo_v2_large", "results.csv"))

# Check if we found the data
if yolo_v1 is None or yolo_v2 is None:
    print("ERROR: Could not find YOLO results. Did you run the training first?")
    exit(1)

# 3. PREPARE DATA FOR PLOTTING
labels = ['V1 Baseline', 'V2 Scaled']
yolo_map50 = [yolo_v1['mAP50'], yolo_v2['mAP50']]
frcnn_map50 = [frcnn_v1['mAP50'], frcnn_v2['mAP50']]

yolo_map_range = [yolo_v1['mAP50-95'], yolo_v2['mAP50-95']]
frcnn_map_range = [frcnn_v1['mAP50-95'], frcnn_v2['mAP50-95']]

# 4. PLOTTING
plt.figure(figsize=(12, 6))

# Plot 1: mAP50 Comparison
plt.subplot(1, 2, 1)
x = range(len(labels))
width = 0.35
plt.bar([i - width/2 for i in x], yolo_map50, width, label='YOLOv11s', color='tab:blue')
plt.bar([i + width/2 for i in x], frcnn_map50, width, label='Faster R-CNN', color='tab:red')
plt.ylabel('mAP @ 0.50')
plt.title('Accuracy Comparison (mAP50)')
plt.xticks(x, labels)
plt.legend()
plt.ylim(0, 1.1)
for i, v in enumerate(yolo_map50): plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
for i, v in enumerate(frcnn_map50): plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')

# Plot 2: mAP50-95 Comparison (Box Quality)
plt.subplot(1, 2, 2)
plt.bar([i - width/2 for i in x], yolo_map_range, width, label='YOLOv11s', color='tab:blue', alpha=0.7)
plt.bar([i + width/2 for i in x], frcnn_map_range, width, label='Faster R-CNN', color='tab:red', alpha=0.7)
plt.ylabel('mAP @ 0.50:0.95')
plt.title('Box Tightness (mAP 50-95)')
plt.xticks(x, labels)
plt.legend()
plt.ylim(0, 1.1)
for i, v in enumerate(yolo_map_range): plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
for i, v in enumerate(frcnn_map_range): plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')

plt.tight_layout()
os.makedirs('final_report_assets', exist_ok=True)
plt.savefig('final_report_assets/comparison_accuracy.png')
print("Comparison plot saved to 'final_report_assets/comparison_accuracy.png'")

# Plot 3: Training Time
plt.figure(figsize=(8, 5))
times_yolo = [yolo_v1['training_time'], yolo_v2['training_time']]
times_frcnn = [frcnn_v1['training_time'], frcnn_v2['training_time']]
plt.bar([i - width/2 for i in x], times_yolo, width, label='YOLOv11s', color='tab:blue')
plt.bar([i + width/2 for i in x], times_frcnn, width, label='Faster R-CNN', color='tab:red')
plt.ylabel('Minutes')
plt.title('Training Efficiency (Total Minutes)')
plt.xticks(x, labels)
plt.legend()
for i, v in enumerate(times_yolo): plt.text(i - width/2, v + 0.1, f'{v:.1f}m', ha='center')
for i, v in enumerate(times_frcnn): plt.text(i + width/2, v + 0.1, f'{v:.1f}m', ha='center')
plt.savefig('final_report_assets/comparison_time.png')
print("Time comparison plot saved to 'final_report_assets/comparison_time.png'")
