import os
import cv2
import torch
from ultralytics import YOLO

def visualize_failures():
    # Load your best trained model
    model_path = 'runs/detect/yolo11_experiment8/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}")
        return

    model = YOLO(model_path)
    
    # Path to validation images
    val_images_path = 'dataset/images/val'
    val_labels_path = 'dataset/labels/val'
    output_dir = 'failure_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Class names from your data.yaml
    class_names = model.names

    print(f"Analyzing validation images in {val_images_path}...")
    
    failures_found = 0
    for img_name in os.listdir(val_images_path):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(val_images_path, img_name)
        label_path = os.path.join(val_labels_path, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # 1. Load Ground Truth (Your labels)
        gt_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    gt_labels.append(cls_id)
        
        # 2. Run Prediction
        results = model.predict(img_path, conf=0.25, verbose=False)[0]
        pred_labels = [int(box.cls) for box in results.boxes]
        
        # 3. Check for mismatches (Simplistic count check or ID check)
        is_failure = False
        if len(gt_labels) != len(pred_labels):
            is_failure = True # Model missed a can or saw a ghost
        else:
            # Check if set of labels matches (ignores location, but good for flavor confusion)
            if sorted(gt_labels) != sorted(pred_labels):
                is_failure = True

        if is_failure:
            failures_found += 1
            # Save the prediction plot to the failure folder
            res_img = results.plot()
            save_path = os.path.join(output_dir, f"fail_{img_name}")
            cv2.imwrite(save_path, res_img)
            print(f"Found mismatch in {img_name}. Saved to {save_path}")
            print(f"  GT Labels: {[class_names[l] for l in gt_labels]}")
            print(f"  Pred Labels: {[class_names[l] for l in pred_labels]}")

    print(f"\nAnalysis complete. Found {failures_found} images with mismatches.")
    print(f"Check the '{output_dir}' folder to see the visual errors.")

if __name__ == "__main__":
    visualize_failures()
