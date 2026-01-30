"""
Processing Thread - Background processing for the GUI.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from process_production import get_timing


class ProcessingThread(QThread):
    """
    Background thread for processing bills.

    Signals:
        progress_updated(current, total, message): Progress update
        result_ready(result_dict): Single bill result ready
        processing_complete(summary_dict): All processing complete
        error_occurred(error_message): Error during processing
    """

    progress_updated = Signal(int, int, str)
    result_ready = Signal(dict)
    processing_complete = Signal(dict)
    error_occurred = Signal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        use_gpu: bool = False,
        verify_pairs: bool = True,
        crop_all: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.verify_pairs = verify_pairs
        self.crop_all = crop_all
        self._stop_requested = False
        self.processor = None  # Will be set during run()

    def run(self):
        """Main processing loop."""
        try:
            # Import processor
            from process_production import ProductionProcessor, Config, ScannerFormatDetector

            # Find config and model
            script_dir = Path(__file__).parent.parent
            config_path = script_dir / "config.yaml"
            patterns_path = script_dir / "patterns_v2.yaml"
            model_path = script_dir / "best.pt"

            if not model_path.exists():
                self.error_occurred.emit(f"YOLO model not found: {model_path}")
                return

            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize processor
            self.progress_updated.emit(0, 0, "Loading models...")

            cfg = Config(config_path if config_path.exists() else None)
            self.processor = ProductionProcessor(
                model_path,
                use_gpu=self.use_gpu,
                cfg=cfg,
                patterns_v2_path=patterns_path if patterns_path.exists() else None
            )

            # Validate directory - check it's not an output directory
            self.progress_updated.emit(0, 0, "Scanning directory...")
            self._validate_input_directory()

            # Detect format and find pairs
            scanner_format, pairs = ScannerFormatDetector.find_pairs(self.input_dir)

            total = len(pairs)
            self.progress_updated.emit(0, total, f"Found {total} bills")

            # Verify pairs if requested
            if self.verify_pairs:
                self.progress_updated.emit(0, total, "Verifying front/back...")
                pairs = self.processor.verify_and_swap_pairs(pairs)

            # Process each pair
            fancy_count = 0
            review_count = 0

            for i, pair in enumerate(pairs):
                if self._stop_requested:
                    break

                timing = get_timing()
                timing.start_bill()

                self.progress_updated.emit(i + 1, total, f"Processing {pair.front_path.name}...")

                # Handle existing errors
                if pair.error:
                    result = {
                        'position': pair.stack_position,
                        'front_file': str(pair.front_path),
                        'back_file': str(pair.back_path) if pair.back_path else '',
                        'serial': '',
                        'fancy_types': '',
                        'confidence': '0.00',
                        'is_fancy': False,
                        'needs_review': True,
                        'serial_region_path': '',
                        'error': pair.error
                    }
                    review_count += 1
                    print(timing.get_summary(f"#{pair.stack_position} ERROR"))
                    self.result_ready.emit(result)
                    continue

                # Extract serial
                serial, confidence, is_upside_down, baseline_variance, star_detected, align_info = self.processor.extract_serial(pair.front_path)
                pair.serial = serial
                pair.confidence = confidence
                pair.is_upside_down = is_upside_down
                pair.baseline_variance = baseline_variance
                pair.star_detected = star_detected
                # Cache alignment info for reuse in generate_crops()
                pair.front_align_angle = align_info.get('angle', 0.0)
                pair.front_align_flipped = align_info.get('flipped', False)

                # Validate
                is_valid, validation_error = self.processor.validate_serial(serial)

                if serial and is_valid:
                    # Check for fancy patterns
                    if self.crop_all:
                        pair.fancy_types = ["ALL"]
                        pair.is_fancy = True
                    else:
                        # Pass baseline_variance in metadata for gas pump detection
                        metadata = {'baseline_variance': pair.baseline_variance}
                        fancy_types = self.processor.pattern_engine.classify_simple(serial, metadata=metadata)
                        pair.fancy_types = fancy_types
                        pair.is_fancy = len(fancy_types) > 0

                    needs_review = confidence < 0.5

                    if pair.is_fancy:
                        fancy_count += 1
                        self.processor.generate_crops(pair, self.output_dir)

                    serial_region_path = ''
                    if needs_review:
                        review_count += 1
                        self.processor._add_to_review_queue(pair, f"Low confidence: {confidence:.2f}", self.output_dir)
                        # Get serial region path from the review item we just added
                        if self.processor.review_queue:
                            serial_region_path = self.processor.review_queue[-1].serial_region_path or ''

                    result = {
                        'position': pair.stack_position,
                        'front_file': str(pair.front_path),
                        'back_file': str(pair.back_path) if pair.back_path else '',
                        'serial': serial,
                        'fancy_types': ', '.join(pair.fancy_types),
                        'confidence': f"{confidence:.2f}",
                        'baseline_variance': f"{pair.baseline_variance:.4f}",
                        'is_fancy': pair.is_fancy,
                        'needs_review': needs_review,
                        'serial_region_path': serial_region_path,
                        'error': ''
                    }
                elif serial and not is_valid:
                    review_count += 1
                    self.processor._add_to_review_queue(pair, f"Validation failed: {validation_error}", self.output_dir)
                    # Get serial region path from the review item we just added
                    serial_region_path = ''
                    if self.processor.review_queue:
                        serial_region_path = self.processor.review_queue[-1].serial_region_path or ''
                    result = {
                        'position': pair.stack_position,
                        'front_file': str(pair.front_path),
                        'back_file': str(pair.back_path) if pair.back_path else '',
                        'serial': serial,
                        'fancy_types': '',
                        'confidence': f"{confidence:.2f}",
                        'baseline_variance': f"{pair.baseline_variance:.4f}",
                        'is_fancy': False,
                        'needs_review': True,
                        'serial_region_path': serial_region_path,
                        'error': validation_error
                    }
                else:
                    review_count += 1
                    self.processor._add_to_review_queue(pair, "No serial detected", self.output_dir)
                    # Get serial region path from the review item we just added
                    serial_region_path = ''
                    if self.processor.review_queue:
                        serial_region_path = self.processor.review_queue[-1].serial_region_path or ''
                    result = {
                        'position': pair.stack_position,
                        'front_file': str(pair.front_path),
                        'back_file': str(pair.back_path) if pair.back_path else '',
                        'serial': '',
                        'fancy_types': '',
                        'confidence': '0.00',
                        'baseline_variance': f"{pair.baseline_variance:.4f}",
                        'is_fancy': False,
                        'needs_review': True,
                        'serial_region_path': serial_region_path,
                        'error': 'No serial detected'
                    }

                # Print timing summary
                bill_id = f"#{pair.stack_position} {result.get('serial') or 'NO_SERIAL'}"
                print(timing.get_summary(bill_id))

                self.result_ready.emit(result)

            # Save review queue
            if self.processor.review_queue:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                review_path = self.input_dir / f"review_queue_{timestamp}.json"
                self.processor.save_review_queue(review_path)

            # Emit completion
            summary = {
                'total': total,
                'fancy_count': fancy_count,
                'review_count': review_count,
                'stopped': self._stop_requested
            }
            self.processing_complete.emit(summary)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

    def request_stop(self):
        """Request the thread to stop."""
        self._stop_requested = True

    def _validate_input_directory(self):
        """Validate the input directory contains scanner images, not cropped output."""
        import re

        files = list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.jpeg"))

        if not files:
            raise ValueError(
                f"No JPEG images found in {self.input_dir}\n\n"
                "Please select a folder containing scanned bill images."
            )

        # Check for cropped output pattern (serial_XX.jpg)
        cropped_pattern = re.compile(r'^[A-L]\d{8}[A-Z*]_\d{2}\.jpe?g$', re.IGNORECASE)
        cropped_count = sum(1 for f in files if cropped_pattern.match(f.name))

        if cropped_count > len(files) * 0.5:  # More than 50% look like cropped output
            raise ValueError(
                f"This directory appears to contain cropped output files, not scanner images.\n\n"
                f"Found {cropped_count} files matching cropped pattern (e.g., B12345678A_01.jpg).\n\n"
                "Please select the original scanner output folder instead."
            )
