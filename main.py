import subprocess
import sys
import time
import os

def run_training(script_name, args, description):
    """
    Runs a training script and waits for it to complete.
    """
    print("\n" + "=" * 80)
    print(f" EXPERIMENT: {description}")
    print(f" COMMAND: {script_name} {' '.join(args)}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    # Use the same python interpreter currently running this script
    try:
        process = subprocess.Popen([sys.executable, script_name] + args)
        # Wait for completion
        process.wait()
        
        end_time = time.time()
        duration = (end_time - start_time) / 60
        
        if process.returncode == 0:
            print(f"\nSUCCESS: {description} completed in {duration:.2f} minutes.")
        else:
            print(f"\nERROR: {description} failed with return code {process.returncode}.")
    except Exception as e:
        print(f"\nCRITICAL ERROR running {description}: {str(e)}")
    
    print("\n" + "#" * 80)
    print(" WAITING 15 SECONDS FOR GPU/RESOURCE COOLDOWN...")
    print("#" * 80 + "\n")
    time.sleep(15)

def main():
    # Define the 4 core experiments
    experiments = [
        # --- YOLO EXPERIMENTS ---
        {
            "script": "train_yolo.py",
            "args": ["--data", "data_v1.yaml", "--name", "yolo_v1_small"],
            "desc": "YOLOv11: BASELINE (Small Dataset v1)"
        },
        {
            "script": "train_yolo.py",
            "args": ["--data", "data_v2.yaml", "--name", "yolo_v2_large"],
            "desc": "YOLOv11: SCALED (Large Dataset v2)"
        },
        
        # --- FASTER R-CNN EXPERIMENTS ---
        {
            "script": "train_frcnn.py",
            "args": [
                "--root", "coco_dataset_v1", 
                "--ann", "coco_dataset_v1/train.json", 
                "--name", "frcnn_v1_small"
            ],
            "desc": "FASTER R-CNN: BASELINE (Small Dataset v1)"
        },
        {
            "script": "train_frcnn.py",
            "args": [
                "--root", "coco_dataset_v2", 
                "--ann", "coco_dataset_v2/train.json", 
                "--name", "frcnn_v2_large"
            ],
            "desc": "FASTER R-CNN: SCALED (Large Dataset v2)"
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
