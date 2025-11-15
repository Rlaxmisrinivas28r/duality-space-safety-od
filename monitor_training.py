# monitor_training.py
import os
import time
import glob

def monitor_training():
    """Monitor the training progress by checking log files"""
    baseline_runs = sorted(glob.glob(r'runs/detect/baseline_run*'), key=os.path.getmtime, reverse=True)
    
    if not baseline_runs:
        print("❌ No training runs found yet")
        return
    
    latest_run = baseline_runs[0]
    print(f"📊 Monitoring: {latest_run}")
    
    # Check for results.csv which contains epoch-by-epoch metrics
    results_file = os.path.join(latest_run, 'results.csv')
    
    while True:
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                
                # Get the last few lines (most recent epochs)
                recent = lines[-min(5, len(lines)):]
                
                if len(lines) > 1:  # Skip header
                    last_epoch_line = lines[-1].strip()
                    print(f"\n✅ Last epoch metrics: {last_epoch_line}")
                    
                    # Count total lines (each line = one epoch)
                    epochs_completed = len(lines) - 1  # -1 for header
                    print(f"📈 Epochs completed: {epochs_completed}/50")
                    
                    if epochs_completed >= 50:
                        print("\n🎉 Training completed!")
                        break
                        
            except Exception as e:
                print(f"⚠️  Error reading results: {e}")
        
        else:
            print(f"⏳ Waiting for training to start... ({results_file})")
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    monitor_training()
