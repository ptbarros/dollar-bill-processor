"""
Monitor Thread - Real-time bill pairing and processing for monitor mode.
"""

import sys
import re
import shutil
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Set

from PySide6.QtCore import QThread, Signal, Slot

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from process_production import get_timing


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

        # Tracking state - FIFO queues for detection-based pairing
        self.front_queue: list[Path] = []  # Queue of unpaired fronts (arrival order)
        self.back_queue: list[Path] = []   # Queue of unpaired backs (arrival order)
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
            'pending_fronts': len(self.front_queue),
            'pending_backs': len(self.back_queue),
        }
        self.processing_complete.emit(summary)

    @Slot(Path)
    def handle_new_file(self, file_path: Path):
        """
        Handle a newly detected file from FileWatcher.

        Called from the main thread via signal connection.
        """
        print(f"[MonitorThread] Received file: {file_path.name}")

        if file_path in self.processed_files:
            print(f"[MonitorThread] Already processed, skipping: {file_path.name}")
            return

        if self.processor is None:
            print(f"[MonitorThread] Processor not ready yet!")
            return

        try:
            print(f"[MonitorThread] Processing {file_path.name}...")

            # Determine if front or back using YOLO detection count
            # Front of bill typically has 6-10 serial region detections
            # Back of bill typically has 0-2 false positive detections
            # Use threshold of 4 to reliably distinguish front from back
            print(f"[MonitorThread] Running YOLO detection...")
            serial_count = self.processor.count_serial_detections(file_path)
            is_front = serial_count >= 4
            print(f"[MonitorThread] Serial count: {serial_count}, is_front: {is_front}")

            self.status_updated.emit(f"Detected: {file_path.name} ({'front' if is_front else 'back'})")

            # Pure detection-based pairing: pair first available front with first available back
            # Ignores filenames entirely - uses YOLO detection + arrival order
            if is_front:
                if self.back_queue:
                    # There's a back waiting - pair with it
                    back_path = self.back_queue.pop(0)
                    print(f"[MonitorThread] Paired: {file_path.name} (front) + {back_path.name} (back)")
                    self._process_pair(file_path, back_path)
                else:
                    # No back waiting - queue this front
                    self.front_queue.append(file_path)
                    print(f"[MonitorThread] Queued front: {file_path.name} (waiting for back, queue: {len(self.front_queue)})")
            else:
                if self.front_queue:
                    # There's a front waiting - pair with it
                    front_path = self.front_queue.pop(0)
                    print(f"[MonitorThread] Paired: {front_path.name} (front) + {file_path.name} (back)")
                    self._process_pair(front_path, file_path)
                else:
                    # No front waiting - queue this back
                    self.back_queue.append(file_path)
                    print(f"[MonitorThread] Queued back: {file_path.name} (waiting for front, queue: {len(self.back_queue)})")

            # Update progress
            total_pending = len(self.front_queue) + len(self.back_queue)
            self.progress_updated.emit(
                self.pair_count,
                self.pair_count + total_pending // 2,
                f"Processed: {self.pair_count} pairs"
            )

        except Exception as e:
            import traceback
            print(f"[MonitorThread] ERROR processing {file_path.name}: {e}")
            traceback.print_exc()
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

        timing = get_timing()
        timing.start_bill()

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
            serial, confidence, is_upside_down, baseline_variance, star_detected, align_info = \
                self.processor.extract_serial(pair.front_path)

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

            # Print timing summary
            bill_id = f"#{position} {serial or 'NO_SERIAL'}"
            print(timing.get_summary(bill_id))

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
            print(timing.get_summary(f"#{position} ERROR"))
            self.error_occurred.emit(f"Error processing pair #{position}: {str(e)}")

    def get_processed_files(self) -> Set[Path]:
        """Get the set of processed files for archiving."""
        return self.processed_files.copy()

    def get_unpaired_files(self) -> Set[Path]:
        """Get all unpaired files (fronts and backs still in queues)."""
        unpaired = set()
        unpaired.update(self.front_queue)
        unpaired.update(self.back_queue)
        return unpaired

    def get_all_session_files(self) -> Set[Path]:
        """Get all files from this session (processed + unpaired)."""
        all_files = self.processed_files.copy()
        all_files.update(self.front_queue)
        all_files.update(self.back_queue)
        return all_files

    def get_all_results(self) -> list:
        """Get all results for export."""
        # Results are emitted via signals; this is for summary export
        return []

    def reset(self):
        """Reset state for a new monitoring session."""
        self.front_queue.clear()
        self.back_queue.clear()
        self.processed_files.clear()
        self.pair_count = 0
        self.fancy_count = 0
        self.review_count = 0
        self._stop_requested = False
