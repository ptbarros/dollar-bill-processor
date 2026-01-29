"""
File Watcher - Directory monitor for incoming scanner files.
"""

import os
import time
from pathlib import Path
from typing import Set

from PySide6.QtCore import QThread, Signal, QFileSystemWatcher


class FileWatcher(QThread):
    """
    Watches a directory for new image files.

    Uses QFileSystemWatcher for immediate notification with polling fallback
    to catch any missed events. Includes file settling logic to ensure files
    are fully written before emitting signals.

    Signals:
        new_file_detected(Path): Emitted when a new, fully-written file is detected
        error_occurred(str): Emitted when an error occurs
    """

    new_file_detected = Signal(Path)
    error_occurred = Signal(str)

    # Supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    def __init__(
        self,
        watch_dir: Path,
        poll_interval: float = 2.0,
        settle_time: float = 1.0,
        parent=None
    ):
        """
        Initialize the file watcher.

        Args:
            watch_dir: Directory to monitor for new files
            poll_interval: Seconds between poll checks (fallback)
            settle_time: Seconds to wait for file to finish writing
            parent: Parent QObject
        """
        super().__init__(parent)
        self.watch_dir = Path(watch_dir)
        self.poll_interval = poll_interval
        self.settle_time = settle_time

        self._stop_requested = False
        self._known_files: Set[Path] = set()
        self._pending_files: dict[Path, float] = {}  # path -> first_seen_time

        # QFileSystemWatcher for immediate notification
        self._fs_watcher = QFileSystemWatcher()
        self._fs_watcher.directoryChanged.connect(self._on_directory_changed)

    def run(self):
        """Main watch loop."""
        # Expand user path (handle ~ on Windows)
        self.watch_dir = self.watch_dir.expanduser().resolve()

        print(f"[FileWatcher] Starting to watch: {self.watch_dir}")

        if not self.watch_dir.exists():
            self.error_occurred.emit(f"Watch directory does not exist: {self.watch_dir}")
            return

        # Initialize known files (don't process existing files)
        self._known_files = self._get_current_files()
        print(f"[FileWatcher] Found {len(self._known_files)} existing files (will be ignored)")

        # Start watching the directory
        success = self._fs_watcher.addPath(str(self.watch_dir))
        print(f"[FileWatcher] QFileSystemWatcher registered: {success}")

        # Main loop with polling fallback
        while not self._stop_requested:
            self._check_for_new_files()
            self._process_pending_files()

            # Sleep in small increments to allow quick stop
            sleep_time = self.poll_interval
            while sleep_time > 0 and not self._stop_requested:
                time.sleep(min(0.1, sleep_time))
                sleep_time -= 0.1

        # Cleanup
        self._fs_watcher.removePath(str(self.watch_dir))

    def stop(self):
        """Request the watcher to stop."""
        self._stop_requested = True

    def _get_current_files(self) -> Set[Path]:
        """Get set of current image files in the watch directory."""
        files = set()
        try:
            for entry in os.scandir(self.watch_dir):
                if entry.is_file():
                    path = Path(entry.path)
                    if path.suffix.lower() in self.IMAGE_EXTENSIONS:
                        files.add(path)
        except OSError as e:
            self.error_occurred.emit(f"Error scanning directory: {e}")
        return files

    def _on_directory_changed(self, path: str):
        """Handle directory change notification from QFileSystemWatcher."""
        # This is called from the main thread, just trigger a check
        self._check_for_new_files()

    def _check_for_new_files(self):
        """Check for new files and add them to pending queue."""
        current_files = self._get_current_files()
        new_files = current_files - self._known_files

        if new_files:
            print(f"[FileWatcher] Detected {len(new_files)} new file(s): {[f.name for f in new_files]}")

        for file_path in new_files:
            if file_path not in self._pending_files:
                self._pending_files[file_path] = time.time()

        self._known_files = current_files

    def _process_pending_files(self):
        """Process pending files that have settled."""
        now = time.time()
        settled_files = []

        for file_path, first_seen in list(self._pending_files.items()):
            # Check if enough time has passed
            if now - first_seen < self.settle_time:
                continue

            # Check if file is ready (stable size)
            if self._is_file_ready(file_path):
                settled_files.append(file_path)
                del self._pending_files[file_path]

        # Emit signals for settled files
        for file_path in sorted(settled_files, key=lambda p: p.name):
            print(f"[FileWatcher] File ready, emitting: {file_path.name}")
            self.new_file_detected.emit(file_path)

    def _is_file_ready(self, file_path: Path) -> bool:
        """
        Check if a file is ready (fully written).

        Uses size stability check - file is ready if size hasn't changed.
        """
        try:
            if not file_path.exists():
                return False

            # Get initial size
            size1 = file_path.stat().st_size
            if size1 == 0:
                return False

            # Wait a short time and check again
            time.sleep(0.1)

            if not file_path.exists():
                return False

            size2 = file_path.stat().st_size

            # File is ready if size is stable
            return size1 == size2

        except OSError:
            return False

    def reset_known_files(self):
        """Reset known files set (call when starting a new batch)."""
        self._known_files = self._get_current_files()
        self._pending_files.clear()
