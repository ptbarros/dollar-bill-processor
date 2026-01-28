"""
Monitor Thread - Real-time bill pairing and processing for monitor mode.
"""

import sys
import re
import shutil
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Set

from PySide6.QtCore import QThread, Signal, Slot

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class MonitorThread(QThread):
    """
    Manages real-time bill pairing and processing during monitor mode.

    Receives file paths from FileWatcher, determines if front/back using YOLO,
    pairs bills based on filename patterns, and processes pairs as they complete.

    Signals (compatible with ProcessingThread):
        progress_updated(current, total, message): Progress update
        result_ready(result_dict): Single bill result ready
        pair_complete(result_dict): Pair finished processing (same as result_ready)
        processing_complete(summary_dict): Called when stop is requested
        error_occurred(error_message): Error during processing
        status_updated(str): Status message for display
    """

    progress_updated = Signal(int, int, str)
    result_ready = Signal(dict)
    pair_complete = Signal(dict)
    processing_complete = Signal(dict)
    error_occurred = Signal(str)
    status_updated = Signal(str)

    def __init__(
        self,
        watch_dir: Path,
        output_dir: Path,
        use_gpu: bool = False,
        verify_pairs: bool = True,
        crop_all: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.verify_pairs = verify_pairs
        self.crop_all = crop_all

        self._stop_requested = False
        self.processor = None

        # Tracking state
        self.pending_fronts: Dict[str, Path] = {}  # base_name -> front_path
        self.pending_backs: Dict[str, Path] = {}   # base_name -> back_path
        self.processed_files: Set[Path] = set()    # Files already processed
        self.pair_count = 0
        self.fancy_count = 0
        self.review_count = 0

    def run(self):
        """Initialize the processor (called when thread starts)."""
        try:
            from process_production import ProductionProcessor, Config

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
            self.status_updated.emit("Loading models...")

            cfg = Config(config_path if config_path.exists() else None)
            self.processor = ProductionProcessor(
                model_path,
                use_gpu=self.use_gpu,
                cfg=cfg,
                patterns_v2_path=patterns_path if patterns_path.exists() else None
            )

            self.status_updated.emit("Monitor ready - waiting for files...")

            # Keep thread alive until stop requested
            while not self._stop_requested:
                self.msleep(100)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

    def stop(self):
        """Request the thread to stop and return summary."""
        self._stop_requested = True

        # Emit completion summary
        summary = {
            'total': self.pair_count,
            'fancy_count': self.fancy_count,
            'review_count': self.review_count,
            'stopped': True,
            'pending_fronts': len(self.pending_fronts),
            'pending_backs': len(self.pending_backs),
        }
        self.processing_complete.emit(summary)

    @Slot(Path)
    def handle_new_file(self, file_path: Path):
        """
        Handle a newly detected file from FileWatcher.

        Called from the main thread via signal connection.
        """
        if file_path in self.processed_files:
            return

        if self.processor is None:
            return

        try:
            # Extract base name for pairing
            base_name = self._get_base_name(file_path)

            # Determine if front or back using YOLO detection count
            # Front of bill has 2 serial numbers, so should have >= 2 detections
            # Back may have 1 false positive, so use threshold of 2
            serial_count = self.processor.count_serial_detections(file_path)
            is_front = serial_count >= 2

            self.status_updated.emit(f"Detected: {file_path.name} ({'front' if is_front else 'back'})")

            if is_front:
                if base_name in self.pending_backs:
                    # Found matching back - process pair
                    back_path = self.pending_backs.pop(base_name)
                    self._process_pair(file_path, back_path)
                else:
                    # Store front, wait for back
                    self.pending_fronts[base_name] = file_path
            else:
                if base_name in self.pending_fronts:
                    # Found matching front - process pair
                    front_path = self.pending_fronts.pop(base_name)
                    self._process_pair(front_path, file_path)
                else:
                    # Store back, wait for front
                    self.pending_backs[base_name] = file_path

            # Update progress
            total_pending = len(self.pending_fronts) + len(self.pending_backs)
            self.progress_updated.emit(
                self.pair_count,
                self.pair_count + total_pending // 2,
                f"Processed: {self.pair_count} pairs"
            )

        except Exception as e:
            self.error_occurred.emit(f"Error processing {file_path.name}: {str(e)}")

    def _get_base_name(self, file_path: Path) -> str:
        """
        Extract base name for pairing.

        Handles various naming conventions:
        - Suffix: 1db_0001.jpg + 1db_0001_b.jpg -> base name "1db_0001"
        - Sequential: 0001.jpg, 0002.jpg -> pairs by number
        """
        name = file_path.stem

        # Check for _b suffix pattern (e.g., 1db_0001_b -> 1db_0001)
        if name.lower().endswith('_b'):
            return name[:-2]

        # For suffix-style naming, the front file IS the base name
        # Check if there's a corresponding _b file pattern
        # Return the stem as-is so it matches when _b version strips suffix
        return name

    def _process_pair(self, front_path: Path, back_path: Path):
        """Process a front/back pair."""
        from process_production import BillPair

        self.pair_count += 1
        position = self.pair_count

        self.status_updated.emit(f"Processing pair #{position}...")

        pair = BillPair(
            front_path=front_path,
            back_path=back_path,
            stack_position=position
        )

        try:
            # Extract serial
            serial, confidence, is_upside_down, baseline_variance, star_detected = \
                self.processor.extract_serial(pair.front_path)

            pair.serial = serial
            pair.confidence = confidence
            pair.is_upside_down = is_upside_down
            pair.baseline_variance = baseline_variance
            pair.star_detected = star_detected

            # Validate
            is_valid, validation_error = self.processor.validate_serial(serial)

            if serial and is_valid:
                # Check for fancy patterns
                if self.crop_all:
                    pair.fancy_types = ["ALL"]
                    pair.is_fancy = True
                else:
                    metadata = {'baseline_variance': pair.baseline_variance}
                    fancy_types = self.processor.pattern_engine.classify_simple(serial, metadata=metadata)
                    pair.fancy_types = fancy_types
                    pair.is_fancy = len(fancy_types) > 0

                needs_review = confidence < 0.5

                if pair.is_fancy:
                    self.fancy_count += 1
                    self.processor.generate_crops(pair, self.output_dir)

                serial_region_path = ''
                if needs_review:
                    self.review_count += 1
                    self.processor._add_to_review_queue(pair, f"Low confidence: {confidence:.2f}", self.output_dir)
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
                self.review_count += 1
                self.processor._add_to_review_queue(pair, f"Validation failed: {validation_error}", self.output_dir)
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
                self.review_count += 1
                self.processor._add_to_review_queue(pair, "No serial detected", self.output_dir)
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

            # Mark files as processed
            self.processed_files.add(front_path)
            if back_path:
                self.processed_files.add(back_path)

            # Emit result
            self.result_ready.emit(result)
            self.pair_complete.emit(result)

            status = f"#{position}: {serial or 'N/A'}"
            if pair.is_fancy:
                status += f" [{', '.join(pair.fancy_types)}]"
            self.status_updated.emit(status)

        except Exception as e:
            result = {
                'position': position,
                'front_file': str(front_path),
                'back_file': str(back_path) if back_path else '',
                'serial': '',
                'fancy_types': '',
                'confidence': '0.00',
                'baseline_variance': '0.0000',
                'is_fancy': False,
                'needs_review': True,
                'serial_region_path': '',
                'error': str(e)
            }
            self.result_ready.emit(result)
            self.error_occurred.emit(f"Error processing pair #{position}: {str(e)}")

    def get_processed_files(self) -> Set[Path]:
        """Get the set of processed files for archiving."""
        return self.processed_files.copy()

    def get_all_results(self) -> list:
        """Get all results for export."""
        # Results are emitted via signals; this is for summary export
        return []

    def reset(self):
        """Reset state for a new monitoring session."""
        self.pending_fronts.clear()
        self.pending_backs.clear()
        self.processed_files.clear()
        self.pair_count = 0
        self.fancy_count = 0
        self.review_count = 0
        self._stop_requested = False
