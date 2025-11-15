# train_v2.py

import os
import glob
import math
from ultralytics import YOLO

try:
    import torch
    has_torch = True
except Exception:
    has_torch = False

def read_config_path(cfg='config.yaml'):
    """Very small parser to get 'path' and 'train' entries from config.yaml"""
    path = None
    train = None
    try:
        with open(cfg, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('path:'):
                    path = line.split(':',1)[1].strip()
                if line.startswith('train:'):
                    train = line.split(':',1)[1].strip()
    except FileNotFoundError:
        return None, None
    return path, train

def count_labels(data_root, train_subdir):
    """Count occurrences per class in label files (YOLO txt format)
    Returns dict[class_id] = count and total files
    """
    labels_dir = os.path.join(data_root, train_subdir, 'labels')
    counts = {}
    total_files = 0
    if not os.path.exists(labels_dir):
        return counts, total_files

    for txt in glob.glob(os.path.join(labels_dir, '*.txt')):
        total_files += 1
        try:
            with open(txt, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = parts[0]
                    counts[cls] = counts.get(cls, 0) + 1
        except Exception:
            continue
    return counts, total_files

def main():
    print("Loading YOLOv8 Nano model for training...")
    model = YOLO('yolov8n.pt')

    # auto-detect device: prefer GPU if available
    device = 'cpu'
    if has_torch and torch.cuda.is_available():
        device = '0'  # Ultralyitcs accepts '0' for first GPU

    # Read data config and compute simple class frequencies
    data_root, train_sub = read_config_path('config.yaml')
    if data_root and train_sub:
        counts, total = count_labels(data_root, train_sub)
        print(f"Dataset found: {total} label files in {os.path.join(data_root, train_sub, 'labels')}")
        if counts:
            print("Class counts (sample):")
            for k,v in sorted(counts.items(), key=lambda x:int(x[0])):
                print(f"  class {k}: {v}")
    else:
        print("Warning: Could not parse config.yaml for dataset path. Proceeding without class counts.")

    # Hyperparameters (conservative defaults)
    epochs = 100
    imgsz = 640
    name = 'baseline_run_v2'

    # Choose batch size depending on device
    if device == 'cpu':
        batch = 8
    else:
        batch = 16

    # Two-stage training: short freeze, then full training (helps stabilization)
    stage1_epochs = max(5, int(math.ceil(0.15 * epochs)))
    stage2_epochs = epochs - stage1_epochs

    print(f"Starting model training on dataset with device={device}, epochs={epochs}, batch={batch}")
    print(f"Stage1: {stage1_epochs} epochs (freeze early layers). Stage2: {stage2_epochs} epochs (unfreeze).")

    # Stage 1: freeze backbone layers to stabilize training (safe default)
    print("--> Stage 1: training with frozen early layers and stronger augmentation")
    model.train(
        data='config.yaml',
        epochs=stage1_epochs,
        imgsz=imgsz,
        name=name + '_stage1',
        device=device,
        batch=batch,
        freeze=10,             # freeze first N layers (Ultralytics accepts this param)
        augment=True,
        auto_augment='randaugment',
        mosaic=1.0,
        mixup=0.2,
        save=True,
        plots=True
    )

    # Stage 2: unfreeze and continue training (resume from last weights)
    print("--> Stage 2: unfreeze and continue training (resume)")
    # resume=True will continue from the last saved checkpoint in the run folder
    model.train(
        data='config.yaml',
        epochs=epochs,
        imgsz=imgsz,
        name=name,
        device=device,
        batch=batch,
        resume=True,
        augment=True,
        auto_augment='randaugment',
        mixup=0.1,
        save=True,
        plots=True
    )

    print("--- Training script finished (check runs/detect/ for outputs) ---")

if __name__ == '__main__':
    main()+5
    