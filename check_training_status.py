# check_training_status.py
import os
import time
import glob
from datetime import datetime

def check_status():
    """Check current training status"""
    baseline_runs = sorted(glob.glob(r'runs/detect/baseline_run*'), key=os.path.getmtime, reverse=True)
    
    if not baseline_runs:
        print("❌ No training runs found")
        return
    
    latest_run = baseline_runs[0]
    run_name = os.path.basename(latest_run)
    print(f"\n{'='*60}")
    print(f"📊 Latest Run: {run_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check weights directory
    weights_dir = os.path.join(latest_run, 'weights')
    if os.path.exists(weights_dir):
        weights = os.listdir(weights_dir)
        print(f"✅ Weights saved: {weights}")
    
    # Check results CSV
    results_file = os.path.join(latest_run, 'results.csv')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        epochs_completed = len(lines) - 1  # -1 for header
        progress = (epochs_completed / 50) * 100
        
        print(f"\n📈 Training Progress: {epochs_completed}/50 epochs ({progress:.1f}%)")
        print(f"{'█' * int(progress/2)}{' ' * (50-int(progress/2))} {progress:.1f}%")
        
        if epochs_completed > 0:
            last_line = lines[-1].strip()
            metrics = last_line.split(',')
            print(f"\n📊 Latest Epoch Metrics (Last line):")
            print(f"   Raw: {last_line[:150]}...")
    else:
        print(f"⏳ Results file not created yet: {results_file}")
        print("   Training may still be loading data...")
    
    # Check training logs
    print(f"\n📂 Run Directory Contents:")
    for item in os.listdir(latest_run):
        item_path = os.path.join(latest_run, item)
        if os.path.isdir(item_path):
            size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                      for dirpath, dirnames, filenames in os.walk(item_path) 
                      for filename in filenames)
            print(f"   📁 {item}/ ({size/(1024*1024):.2f} MB)")
        else:
            size = os.path.getsize(item_path)
            print(f"   📄 {item} ({size/1024:.2f} KB)")
    
    print(f"\n✅ All files being saved to: {latest_run}")

if __name__ == "__main__":
    check_status()
