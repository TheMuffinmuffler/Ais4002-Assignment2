import polars as pl
import os


if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')

def get_best_yolo(csv_path):
    if not os.path.exists(csv_path): return {"mAP50": 0, "mAP50-95": 0, "time": 0}
    df = pl.read_csv(csv_path)
    df = df.rename({c: c.strip() for c in df.columns})
    
    # Get the best mAP50
    best_row = df.filter(pl.col("metrics/mAP50(B)") == df["metrics/mAP50(B)"].max()).row(0, named=True)
    
    # YOLO 'time' column is cumulative, so the last row's value is the total time in seconds
    total_time_min = df["time"].max() / 60
    
    return {"mAP50": best_row["metrics/mAP50(B)"], "mAP50-95": best_row["metrics/mAP50-95(B)"], "time": total_time_min}

# Data Gathering
yolo_v1 = get_best_yolo(os.path.join("runs", "detect", "yolo_v1_small", "results.csv"))
yolo_v2 = get_best_yolo(os.path.join("runs", "detect", "yolo_v2_large", "results.csv"))

# FRCNN Results from validated logs
frcnn_v1 = {"mAP50": 0.564, "mAP50-95": 0.457, "time": 4.06}
frcnn_v2 = {"mAP50": 0.957, "mAP50-95": 0.771, "time": 9.56}

# Synthesis Table
print("\n" + "="*85)
print(f"{'Model Experiment':<35} | {'mAP50':<8} | {'mAP50-95':<10} | {'Train Time':<12}")
print("-" * 85)
print(f"{'YOLOv11s: Baseline (V1)':<35} | {yolo_v1['mAP50']:<8.3f} | {yolo_v1['mAP50-95']:<10.3f} | {yolo_v1['time']:<12.2f} min")
print(f"{'YOLOv11s: Scaled (V2)':<35} | {yolo_v2['mAP50']:<8.3f} | {yolo_v2['mAP50-95']:<10.3f} | {yolo_v2['time']:<12.2f} min")
print(f"{'FRCNN: Baseline (V1)':<35} | {frcnn_v1['mAP50']:<8.3f} | {frcnn_v1['mAP50-95']:<10.3f} | {frcnn_v1['time']:<12.2f} min")
print(f"{'FRCNN: Scaled (V2)':<35} | {frcnn_v2['mAP50']:<8.3f} | {frcnn_v2['mAP50-95']:<10.3f} | {frcnn_v2['time']:<12.2f} min")
print("="*85)

# Calculate Impact of Scaling (V1 to V2)
yolo_gain = ((yolo_v2['mAP50'] - yolo_v1['mAP50']) / yolo_v1['mAP50']) * 100 if yolo_v1['mAP50'] > 0 else 0
frcnn_gain = ((frcnn_v2['mAP50'] - frcnn_v1['mAP50']) / frcnn_v1['mAP50']) * 100

print("\nSCALING ANALYSIS (Impact of adding more data):")
print(f" - YOLOv11s Improvement:  +{yolo_gain:.1f}% mAP50")
print(f" - Faster R-CNN Improvement: +{frcnn_gain:.1f}% mAP50")
print("\nCONCLUSION FOR REPORT:")
if frcnn_gain > yolo_gain:
    print("Faster R-CNN benefited significantly more from the extra data, proving it is a 'data-hungry' model.")
    print("YOLO was already high-performing at low data, showing better architectural efficiency.")
