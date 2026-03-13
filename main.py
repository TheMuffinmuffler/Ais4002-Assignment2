import subprocess
import sys
import time
import os

def run_training(script_name, args, description):
    """
    Runs a training script from the src folder.
    """
    script_path = os.path.join("src", script_name)
    print("\n" + "=" * 80)
    print(f" EXPERIMENT: {description}")
    print(f" COMMAND: python {script_path} {' '.join(args)}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    try:
        # We specify the current directory as the project root so scripts find 'data/' correctly
        process = subprocess.Popen([sys.executable, script_path] + args)
        process.wait()
        
        duration = (time.time() - start_time) / 60
        if process.returncode == 0:
            print(f"\nSUCCESS: {description} completed in {duration:.2f} minutes.")
        else:
            print(f"\nERROR: {description} failed.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
    
    time.sleep(5)

def main():
    # Update paths to point to the new 'data' directory
    experiments = [
        {
            "script": "train_yolo.py",
            "args": ["--data", "data_v1.yaml", "--name", "yolo_v1_small"],
            "desc": "YOLOv11: BASELINE (V1)"
        },
        {
            "script": "train_yolo.py",
            "args": ["--data", "data_v2.yaml", "--name", "yolo_v2_large"],
            "desc": "YOLOv11: SCALED (V2)"
        },
        {
            "script": "train_frcnn.py",
            "args": [
                "--root", os.path.join("data", "coco_dataset_v1"), 
                "--train_ann", os.path.join("data", "coco_dataset_v1", "train.json"), 
                "--val_ann", os.path.join("data", "coco_dataset_v1", "val.json"),
                "--name", "frcnn_v1_validated"
            ],
            "desc": "FASTER R-CNN: BASELINE (V1)"
        },
        {
            "script": "train_frcnn.py",
            "args": [
                "--root", os.path.join("data", "coco_dataset_v2"), 
                "--train_ann", os.path.join("data", "coco_dataset_v2", "train.json"), 
                "--val_ann", os.path.join("data", "coco_dataset_v2", "val.json"),
                "--name", "frcnn_v2_validated"
            ],
            "desc": "FASTER R-CNN: SCALED (V2)"
        }
    ]

    print("\n" + "!" * 80)
    print(" STARTING MODEL COMPARISON EXPERIMENT SERIES")
    print(f" TOTAL TASKS: {len(experiments)}")
    print("!" * 80 + "\n")

    for exp in experiments:
        run_training(exp["script"], exp["args"], exp["desc"])

    print("\n" + "=" * 80)
    print(" ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print("\nRESULTS DIRECTORIES:")
    print("  - YOLO v1: runs/detect/yolo_v1_small")
    print("  - YOLO v2: runs/detect/yolo_v2_large")
    print("  - FRCNN v1: runs/frcnn/frcnn_v1_small")
    print("  - FRCNN v2: runs/frcnn/frcnn_v2_large")
    print("\nTIP: Compare the 'best.pt' and 'best.pth' weights in these folders.")

if __name__ == "__main__":
    main()
