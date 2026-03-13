from ultralytics import YOLO
import cv2
import os
import glob

# Identify the best models
yolo_model_path = "runs/detect/yolo_v2_large4/weights/best.pt"
val_images_path = "yolo_dataset_v2/images/val/*.jpg" # or whatever your images are
output_dir = "failure_analysis"
os.makedirs(output_dir, exist_ok=True)

print("\nRunning Failure Analysis for YOLO...")
model = YOLO(yolo_model_path)
images = glob.glob(val_images_path)

# Limit to first 20 to avoid over-processing
for img_path in images[:20]:
    results = model(img_path, verbose=False)
    for r in results:
        # Check if model has low confidence detections (< 0.6)
        low_conf = [box for box in r.boxes if box.conf < 0.6]
        if low_conf:
            name = os.path.basename(img_path)
            r.save(filename=os.path.join(output_dir, f"yolo_low_conf_{name}"))
            print(f" - Saved low confidence example: {name}")

print("\nFailure Analysis Complete.")
print(f"Check the '{output_dir}' folder for images where the model was least certain.")
