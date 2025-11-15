# train.py
from ultralytics import YOLO

# 1. Load the Model
print("Loading YOLOv8 Nano model for training...")
model = YOLO('yolov8n.pt') 

# 2. Start training the model [cite: 59]
# The name='baseline_run' creates the folder where all results go.
print("Starting model training on the Duality AI dataset...")
results = model.train(
    data='config.yaml', 
    epochs=50, 
    imgsz=640, 
    name='baseline_run'
)

print("--- Training Complete ---")
print("Your next step is to run the predict.py script.")