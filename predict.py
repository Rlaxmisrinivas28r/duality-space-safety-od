# predict.py
from ultralytics import YOLO

# 1. Define the Path to Your Trained Weights
# This path points to the 'best.pt' file created by train.py inside the runs folder.
weights_path = 'runs/detect/baseline_run/weights/best.pt'
print(f"Loading best model weights from: {weights_path}")

try:
    model = YOLO(weights_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Training must be completed successfully before running prediction.")
    exit()

# 2. Run Validation/Evaluation on the test set [cite: 147]
print("Starting final model evaluation on the test set...")
metrics = model.val(
    data='config.yaml', 
    split='test' 
)

# 3. Print Key Metrics for your Report
print("\n--- Model Performance Metrics ---")
print(f"Primary Score (mAP@0.5): {metrics.box.map50}") # Your score for the 80 points [cite: 85, 87]
print("---------------------------------")
print(f"Full results (Confusion Matrix, graphs) saved in: {metrics.save_dir}")