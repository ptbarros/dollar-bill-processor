# GEMINI.md

## Project Overview

This project is a production-ready system for automatically processing scanned US dollar bills to identify "fancy" serial numbers (e.g., repeaters, radars, low numbers) and prepare them for eBay listings.

It uses a hybrid computer vision pipeline:
1.  **Template Alignment:** An initial pass using OpenCV's ORB feature detection aligns the scanned bill with a reference image (`reference_bill.jpg`) to correct for positional drift and rotation.
2.  **YOLOv8 Detection:** A trained YOLOv8 Nano model (`best.pt`) detects the bounding boxes of the two serial numbers on the aligned bill.
3.  **EasyOCR Extraction:** The detected bounding boxes are expanded slightly and fed into EasyOCR to extract the serial number text.
4.  **Fancy Number Classification:** The extracted serial number is checked against a set of predefined patterns (solid, repeater, radar, ladder, low serial, binary, star notes).
5.  **CSV Output:** The results, including the filename, serial number, fancy types, confidence, and processing time, are saved to a timestamped CSV file.

The system is optimized for batch processing and can achieve a rate of approximately 176 bills per minute on a standard CPU.

## Building and Running

### Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Processor

There are two ways to run the main processing script:

1.  **Using the shell script (recommended for default settings):**
    This will process all `Dollar_*.jpg` files in the project's root directory.
    ```bash
    ./run_yolo.sh
    ```

2.  **Running the Python script directly:**
    This allows you to specify a target directory for the scanned images.

    ```bash
    # Process bills in a specific directory
    python process_bills_yolo.py best.pt /path/to/your/scans

    # Process bills in the current directory
    python process_bills_yolo.py best.pt .
    ```

### Output

The script generates a CSV file named `serial_numbers_YYYYMMDD_HHMMSS.csv` in the directory where the images were processed.

## Training the Model

The YOLOv8 model can be retrained using the provided `train_yolo.py` script. The training data is expected to be in the `yolov8/` directory, following the structure defined in `yolov8/data_local.yaml`.

```bash
# Train for 100 epochs with the nano model on CPU (default)
python train_yolo.py

# Train for 50 epochs
python train_yolo.py 50

# Train with the 'small' model
python train_yolo.py 100 s

# Train on a GPU
python train_yolo.py 100 n gpu
```

The newly trained model will be saved as `best.pt` in the project root.

## Development Conventions

*   **Branching:** Development for new features should be done on a feature branch (e.g., `feature/production-pipeline`). The `main` branch should remain stable.
*   **Commit Messages:** Follow a conventional commit format (e.g., `feat:`, `fix:`, `docs:`).
*   **Code Style:** The code is well-structured into classes. Follow the existing patterns for new development.
*   **GPU Usage:** GPU acceleration is disabled by default. To enable it, you must edit `process_bills_yolo.py` and set `use_gpu=True` in the `BillProcessor` instantiation.
