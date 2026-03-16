import os
import shutil


if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')

source_dir = "runs/detect/yolo_v2_large"
target_dir = "final_report_assets"
os.makedirs(target_dir, exist_ok=True)

files_to_copy = [
    "BoxPR_curve.png",
    "confusion_matrix.png",
    "results.png",
    "val_batch0_pred.jpg"
]

print("\nCollecting YOLO plots...")
for f in files_to_copy:
    src = os.path.join(source_dir, f)
    dst = os.path.join(target_dir, f"yolo_{f}")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f" - Copied: {f} -> final_report_assets/yolo_{f}")
    else:
        print(f" - SKIP (not found): {src}")

print("\nAll plots collected in 'final_report_assets/'.")
