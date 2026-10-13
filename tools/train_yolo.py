#!/usr/bin/env python3
"""
Train YOLOv8 model for dollar bill detection (Dollar Detective).

This script trains a YOLOv8 Nano model on your labeled dataset.
Detects 10 classes: serial_number, bill_front, bill_back, front_plate,
back_plate, seal_f, seal_t, denomination, series_year, star_symbol.

Training typically takes 5-15 minutes on CPU, <2 minutes on GPU.

Output:
  - best.pt (best model during training)
  - last.pt (final model)
  - results.png (training metrics)
"""

from ultralytics import YOLO
from pathlib import Path
import sys

# Dataset location - update this path when switching datasets
DATASET_PATH = Path("/home/pbarros/projects/dataset1/data_local.yaml")


def train_model(data_yaml, epochs=100, imgsz=640, model_size='n', use_gpu=False):
    """
    Train YOLOv8 model.

    Args:
        data_yaml: Path to data.yaml config file
        epochs: Number of training epochs (default: 100)
        imgsz: Image size for training (default: 640)
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium)
        use_gpu: Whether to use GPU for training
    """

    # Model sizes and descriptions
    model_info = {
        'n': ('yolov8n.pt', 'Nano - Fastest, smallest'),
        's': ('yolov8s.pt', 'Small - Balanced'),
        'm': ('yolov8m.pt', 'Medium - More accurate, slower'),
    }

    model_file, description = model_info.get(model_size, model_info['n'])

    print("="*70)
    print("YOLOv8 Training for Dollar Detective (Multi-Label)")
    print("="*70)
    print(f"Model: {model_file} ({description})")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Device: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print("="*70)
    print()

    # Load a pretrained YOLO model
    model = YOLO(model_file)

    # Train the model
    print("Starting training...")
    print("This will take 5-15 minutes on CPU, <2 minutes on GPU")
    print()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        patience=20,  # Early stopping if no improvement for 20 epochs
        save=True,
        device='0' if use_gpu else 'cpu',
        plots=True,
        verbose=True
    )

    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)

    # Get the path to the trained model
    save_dir = Path(model.trainer.save_dir)
    best_model = save_dir / 'weights' / 'best.pt'
    last_model = save_dir / 'weights' / 'last.pt'

    print(f"\nBest model saved to: {best_model}")
    print(f"Last model saved to: {last_model}")
    print(f"Training results: {save_dir / 'results.png'}")
    print()
    print("To use your model:")
    print(f"  cp {best_model} ../best.pt")
    print(f"  cd ..")
    print(f"  ./run_yolo.sh")
    print()

    return best_model, last_model


def main():
    """Main entry point."""

    # Default configuration - uses DATASET_PATH constant defined at top of file
    data_yaml = DATASET_PATH

    # Check if data.yaml exists
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found!")
        print("\nMake sure you have the YOLOv8 dataset in the 'yolov8' folder")
        return

    # Parse command line arguments
    epochs = 100  # Default
    model_size = 'n'  # Nano (fastest)
    use_gpu = False

    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    if len(sys.argv) > 2:
        model_size = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3].lower() == 'gpu':
        use_gpu = True

    # Train the model
    best_model, last_model = train_model(
        str(data_yaml),
        epochs=epochs,
        model_size=model_size,
        use_gpu=use_gpu
    )

    # Copy best model to main folder for easy access
    import shutil
    dest = Path(__file__).parent / 'best.pt'
    shutil.copy(best_model, dest)
    print(f"✓ Copied best model to: {dest}")
    print()
    print("Ready to process bills! Run:")
    print("  ./run_yolo.sh")


if __name__ == "__main__":
    print("\nUsage:")
    print("  python train_yolo.py [epochs] [model_size] [gpu]")
    print()
    print("Examples:")
    print("  python train_yolo.py              # Train for 100 epochs on CPU")
    print("  python train_yolo.py 50           # Train for 50 epochs on CPU")
    print("  python train_yolo.py 100 s        # Train 'small' model")
    print("  python train_yolo.py 100 n gpu    # Train on GPU")
    print()
    input("Press Enter to start training with defaults (100 epochs, nano model, CPU)...")
    print()
    main()
