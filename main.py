import subprocess
import sys
import time

def run_training(script_name, description):
    """
    Runs a training script and waits for it to complete.
    """
    print("=" * 60)
    print(f"STARTING: {description}")
    print(f"Running script: {script_name}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Use the same python interpreter currently running this script
    process = subprocess.Popen([sys.executable, script_name])
    
    # Wait for completion
    process.wait()
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    
    if process.returncode == 0:
        print(f"\nSUCCESS: {description} completed in {duration:.2f} minutes.")
    else:
        print(f"\nERROR: {description} failed with return code {process.returncode}.")
        sys.exit(process.returncode)

def main():
    # 1. Train YOLOv11
    # Note: 100 epochs as defined in train_yolo.py
    run_training("train_yolo.py", "YOLOv11 Training (100 Epochs)")
    
    print("\n" + "#" * 60)
    print("WAITING 15 SECONDS FOR GPU MEMORY TO CLEAR...")
    print("#" * 60 + "\n")
    time.sleep(15)
    
    # 2. Train Faster R-CNN
    # Note: 10 epochs as defined in train_frcnn.py
    run_training("train_frcnn.py", "Faster R-CNN Training (10 Epochs)")

    print("\n" + "=" * 60)
    print("ALL TRAINING SESSIONS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
