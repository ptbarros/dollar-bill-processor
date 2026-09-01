"""
Processing Thread - Background processing for the GUI.
"""

import sys
import cv2
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from process_production import get_timing
from debug_logger import dlog_raw


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
        auto_crop: bool = True,
        extract_plate_info: bool = False,
        debug_logging: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.debug_logging = debug_logging
        self.verify_pairs = verify_pairs
        self.crop_all = crop_all
        self.auto_crop = auto_crop
        self.extract_plate_info = extract_plate_info
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
            patterns_dir = script_dir / "patterns"
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
                patterns_dir=patterns_dir if patterns_dir.exists() else None
            )

            if self.debug_logging:
                if getattr(self.processor, 'use_onnx', False):
                    provs = getattr(self.processor.yolo_model, 'providers', [])
                    backend = f"ONNX Runtime {list(provs)}"
                else:
                    backend = "torch/ultralytics"
                dlog_raw(f"[BACKEND] {backend} | gpu_acceleration={self.use_gpu}")

            # Validate directory - check it's not an output directory
            self.progress_updated.emit(0, 0, "Scanning directory...")
            self._validate_input_directory()

            # Detect format and find pairs
            scanner_format, pairs = ScannerFormatDetector.find_pairs(self.input_dir)

            total = len(pairs)
            self.progress_updated.emit(0, total, f"Found {total} bills ({scanner_format} format)")

            # Batch timing
            batch_start = time.time()
            verify_time = 0.0
            verify_yolo_calls = 0
            total_yolo_calls = 0
            total_ocr_calls = 0

            # Verify pairs if requested, unless pre-organized format detected
            # dollar_sequential files are already verified and oriented by bill wizard
            if self.verify_pairs and scanner_format != 'dollar_sequential':
                verify_start = time.time()
                verify_yolo_start = get_timing().yolo_calls  # Track YOLO calls during verify
                def verify_progress(current, verify_total):
                    self.progress_updated.emit(current, verify_total, f"Verifying pairs: {current}/{verify_total}")
                pairs = self.processor.verify_and_swap_pairs(pairs, progress_callback=verify_progress)
                verify_time = time.time() - verify_start
                verify_yolo_calls = get_timing().yolo_calls - verify_yolo_start
                verify_line = f"[VERIFY] {total} pairs verified in {verify_time:.2f}s ({verify_yolo_calls} YOLO calls, {verify_time/total:.3f}s/pair)"
                print(verify_line)
                if self.debug_logging:
                    dlog_raw(verify_line)

            # Process each pair
            fancy_count = 0
            review_count = 0

            for i, pair in enumerate(pairs):
                if self._stop_requested:
                    break

                timing = get_timing()
                timing.start_bill()  # This resets yolo_calls/ocr_calls to 0

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
                        'error': pair.error,
                        'front_align_angle': 0.0,
                        'front_align_flipped': False,
                        'series_year': '',
                        'front_plate': '',
                        'back_plate': '',
                        'potential_mule': False,
                        'serial_mismatch': False,
                    }
                    review_count += 1
                    print(timing.get_summary(f"#{pair.stack_position} ERROR"))
                    self.result_ready.emit(result)
                    continue

                # Extract serial, using cached detections if available from verify stage
                # For pre-organized dollar_sequential format, skip YOLO alignment
                pre_aligned = (scanner_format == 'dollar_sequential')
                serial, confidence, is_upside_down, baseline_variance, star_detected, align_info = self.processor.extract_serial(
                    pair.front_path, cached_detections=pair.front_cache, pre_aligned=pre_aligned)

                # Lazy detection: if no serial found and we have a back image, swap and retry
                # This only runs when verify_pairs is disabled (otherwise pairs are pre-verified)
                if not serial and pair.back_path and not self.verify_pairs and not pre_aligned:
                    # Swap front and back
                    pair.front_path, pair.back_path = pair.back_path, pair.front_path
                    pair.front_cache, pair.back_cache = pair.back_cache, pair.front_cache
                    pair.swapped = True
                    # Retry serial extraction on the swapped front
                    serial, confidence, is_upside_down, baseline_variance, star_detected, align_info = self.processor.extract_serial(
                        pair.front_path, cached_detections=pair.front_cache)

                pair.serial = serial
                pair.confidence = confidence
                pair.is_upside_down = is_upside_down
                pair.baseline_variance = baseline_variance
                pair.star_detected = star_detected
                # Cache alignment info for reuse in generate_crops()
                pair.front_align_angle = align_info.get('angle', 0.0)
                pair.front_align_flipped = align_info.get('flipped', False)
                pair.serial_mismatch = align_info.get('serial_mismatch', False)

                # Cache detection data for later use (plate extraction) if not already cached
                if pair.front_cache is None and '_detections' in align_info:
                    pair.front_cache = align_info['_detections']

                # Calculate overprint shift from aligned front image
                aligned_front = align_info.get('aligned_image')
                if aligned_front is None:
                    aligned_front = cv2.imread(str(pair.front_path))
                shift_x, shift_y, containment = self.processor._calculate_seal_shift(aligned_front)
                pair.seal_shift_x = shift_x
                pair.seal_shift_y = shift_y
                pair.seal_containment = containment

                # Extract plate info if setting enabled
                plate_info = {'series_year': '', 'front_plate': '', 'back_plate': '', 'potential_mule': False}
                if self.extract_plate_info and serial:
                    # Load and align front image, using cached detections if available
                    front_aligned, _ = self.processor.yolo_aligner.align_image(
                        pair.front_path, cached_detections=pair.front_cache)
                    # Load and align back image (for back_plate)
                    back_aligned = None
                    if pair.back_path:
                        back_aligned, _ = self.processor.yolo_aligner.align_image(
                            pair.back_path, cached_detections=pair.back_cache)
                    plate_info = self.processor._extract_plate_info(front_aligned, back_aligned)
                pair.series_year = plate_info['series_year']
                pair.front_plate = plate_info['front_plate']
                pair.back_plate = plate_info['back_plate']
                pair.potential_mule = plate_info.get('potential_mule', False)

                # Validate
                is_valid, validation_error = self.processor.validate_serial(serial)

                if serial and is_valid:
                    # Check for fancy patterns
                    if self.crop_all:
                        pair.fancy_types = ["ALL"]
                        pair.is_fancy = True
                    else:
                        # Pass baseline_variance, seal position, and plate info in metadata
                        metadata = {
                            'baseline_variance': pair.baseline_variance,
                            'gas_pump_threshold': self.processor.pattern_engine.get_gas_pump_threshold(),
                            'seal_x': pair.seal_shift_x,
                            'seal_y': pair.seal_shift_y,
                            'seal_containment': pair.seal_containment,
                            'series_year': pair.series_year,
                            'front_plate': pair.front_plate,
                            'back_plate': pair.back_plate,
                        }
                        fancy_types = self.processor.pattern_engine.classify_simple(serial, metadata=metadata)
                        pair.fancy_types = fancy_types
                        pair.is_fancy = len(fancy_types) > 0

                    needs_review = confidence < 0.5 or pair.serial_mismatch

                    if pair.is_fancy:
                        fancy_count += 1
                        if self.auto_crop:
                            self.processor.generate_crops(pair, self.output_dir)

                    serial_region_path = ''
                    if needs_review:
                        review_count += 1
                        if pair.serial_mismatch:
                            review_reason = "Mismatched serial numbers"
                        else:
                            review_reason = f"Low confidence: {confidence:.2f}"
                        self.processor._add_to_review_queue(pair, review_reason, self.output_dir)
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
                        'baseline_variance': f"{pair.baseline_variance:.1f}",
                        'seal_x': f"{pair.seal_shift_x:.1f}",
                        'seal_y': f"{pair.seal_shift_y:.1f}",
                        'seal_containment': f"{pair.seal_containment:.1f}",
                        'is_fancy': pair.is_fancy,
                        'needs_review': needs_review,
                        'serial_region_path': serial_region_path,
                        'error': '',
                        'front_align_angle': pair.front_align_angle,
                        'front_align_flipped': pair.front_align_flipped,
                        'series_year': pair.series_year,
                        'front_plate': pair.front_plate,
                        'back_plate': pair.back_plate,
                        'potential_mule': pair.potential_mule,
                        'serial_mismatch': pair.serial_mismatch,
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
                        'baseline_variance': f"{pair.baseline_variance:.1f}",
                        'seal_x': f"{pair.seal_shift_x:.1f}",
                        'seal_y': f"{pair.seal_shift_y:.1f}",
                        'seal_containment': f"{pair.seal_containment:.1f}",
                        'is_fancy': False,
                        'needs_review': True,
                        'serial_region_path': serial_region_path,
                        'error': validation_error,
                        'front_align_angle': pair.front_align_angle,
                        'front_align_flipped': pair.front_align_flipped,
                        'series_year': pair.series_year,
                        'front_plate': pair.front_plate,
                        'back_plate': pair.back_plate,
                        'potential_mule': pair.potential_mule,
                        'serial_mismatch': pair.serial_mismatch,
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
                        'baseline_variance': f"{pair.baseline_variance:.1f}",
                        'seal_x': f"{pair.seal_shift_x:.1f}",
                        'seal_y': f"{pair.seal_shift_y:.1f}",
                        'seal_containment': f"{pair.seal_containment:.1f}",
                        'is_fancy': False,
                        'needs_review': True,
                        'serial_region_path': serial_region_path,
                        'error': 'No serial detected',
                        'front_align_angle': pair.front_align_angle,
                        'front_align_flipped': pair.front_align_flipped,
                        'series_year': pair.series_year,
                        'front_plate': pair.front_plate,
                        'back_plate': pair.back_plate,
                        'potential_mule': pair.potential_mule,
                        'serial_mismatch': pair.serial_mismatch,
                    }

                # Print timing summary and accumulate totals
                bill_id = f"#{pair.stack_position} {result.get('serial') or 'NO_SERIAL'}"
                timing_line = timing.get_summary(bill_id)
                print(timing_line)
                if self.debug_logging and timing_line:
                    dlog_raw(timing_line)
                total_yolo_calls += timing.yolo_calls  # Already reset to 0 at start of each bill
                total_ocr_calls += timing.ocr_calls

                self.result_ready.emit(result)

            # Print batch summary
            batch_time = time.time() - batch_start
            processing_time = batch_time - verify_time
            avg_per_bill = processing_time / total if total > 0 else 0
            all_yolo = verify_yolo_calls + total_yolo_calls
            rate = (total / processing_time * 60) if processing_time > 0 else 0
            if getattr(self.processor, 'use_onnx', False):
                backend = f"ONNX Runtime {list(getattr(self.processor.yolo_model, 'providers', []))}"
            else:
                backend = "torch/ultralytics"
            summary_block = "\n".join([
                f"\n{'='*70}",
                f"[BATCH SUMMARY]",
                f"  Backend: {backend} (gpu_acceleration={self.use_gpu})",
                f"  Bills: {total} | Fancy: {fancy_count} | Review: {review_count}",
                f"  Verify: {'ON' if self.verify_pairs else 'OFF'} ({verify_time:.2f}s, {verify_yolo_calls} YOLO)",
                f"  Processing: {processing_time:.2f}s ({avg_per_bill:.2f}s/bill avg)",
                f"  Rate: {rate:.1f} bills/minute",
                f"  Total: {batch_time:.2f}s | YOLO: {all_yolo} | OCR: {total_ocr_calls}",
                f"{'='*70}\n",
            ])
            print(summary_block)
            if self.debug_logging:
                dlog_raw(summary_block)

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


class OrganizeThread(QThread):
    """
    Background thread for organizing bill images.

    Pre-processes images: classifies front/back, fixes orientation, deskews, renames.

    Signals:
        progress_updated(current, total, message): Progress update
        organize_complete(result_dict): Organization complete
        error_occurred(error_message): Error during organization
    """

    progress_updated = Signal(int, int, str)
    organize_complete = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, input_dir: str, use_gpu: bool = False, parent=None):
        super().__init__(parent)
        self.input_dir = Path(input_dir)
        self.use_gpu = use_gpu
        self._stop_requested = False

    def run(self):
        """Main organization loop."""
        try:
            from process_production import ProductionProcessor, Config

            # Find config and model
            script_dir = Path(__file__).parent.parent
            config_path = script_dir / "config.yaml"
            patterns_dir = script_dir / "patterns"
            model_path = script_dir / "best.pt"

            if not model_path.exists():
                self.error_occurred.emit(f"YOLO model not found: {model_path}")
                return

            # Initialize processor
            self.progress_updated.emit(0, 0, "Loading models...")

            cfg = Config(config_path if config_path.exists() else None)
            processor = ProductionProcessor(
                model_path,
                use_gpu=self.use_gpu,
                cfg=cfg,
                patterns_dir=patterns_dir if patterns_dir.exists() else None
            )

            # Run organization
            def progress_callback(current, total, message):
                self.progress_updated.emit(current, total, message)

            result = processor.organize_folder(self.input_dir, progress_callback=progress_callback)

            if 'error' in result:
                self.error_occurred.emit(result['error'])
            else:
                # Print summary
                print(f"\n{'='*70}")
                print(f"[ORGANIZE COMPLETE]")
                print(f"  Pairs organized: {result['pairs_organized']}")
                print(f"  Images corrected: {result['images_corrected']}")
                print(f"  Time taken: {result['time_taken']:.2f}s")
                print(f"  Output format: Dollar_001.jpg through Dollar_{result['pairs_organized']*2:03d}.jpg")
                print(f"{'='*70}\n")

                self.organize_complete.emit(result)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

    def request_stop(self):
        """Request the thread to stop."""
        self._stop_requested = True
