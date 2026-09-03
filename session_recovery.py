"""
Session Recovery Manager
Handles autosave and recovery of session state for crash protection.

Features:
- Periodic autosave of session state to recovery file
- Atomic writes (temp file + rename) for data integrity
- Recovery detection and restoration on startup
- Automatic cleanup after successful archive
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionRecoveryManager:
    """
    Manages session state persistence for crash recovery.

    Usage:
        recovery = SessionRecoveryManager()

        # Save session periodically
        recovery.save_session(results, input_dir, processing_complete=True)

        # On startup, check for recovery
        if recovery.has_recovery_file():
            info = recovery.get_recovery_info()
            # Show dialog to user...
            if user_wants_restore:
                data = recovery.load_recovery()
                # Restore session...
            else:
                recovery.clear_recovery()

        # After successful archive
        recovery.clear_recovery()
    """

    VERSION = 1

    def __init__(self, path: Optional[Path] = None):
        """Initialize the recovery manager.

        Args:
            path: Custom path for recovery file. Defaults to the writable
                per-user data dir (repo root in dev, user config dir when frozen).
        """
        from resource_path import user_data_dir
        self.path = path or (user_data_dir() / ".session_recovery.json")
        self._dirty = False
        self._last_save_hash: Optional[int] = None

    def save_session(
        self,
        results: List[Dict[str, Any]],
        input_directory: str,
        processing_complete: bool = False,
        total_processed: int = 0,
        last_selected_index: int = -1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save current session state to recovery file.

        Uses atomic write (temp file + rename) for data integrity.

        Args:
            results: List of result dictionaries from processing
            input_directory: Path to input directory being processed
            processing_complete: Whether batch processing has finished
            total_processed: Total number of bills processed
            last_selected_index: Index of last selected item in results list
            metadata: Optional additional metadata to store

        Returns:
            True if save was successful, False otherwise
        """
        # Build recovery data structure
        data = {
            "version": self.VERSION,
            "timestamp": datetime.now().isoformat(),
            "input_directory": input_directory,
            "results": results,
            "processing_complete": processing_complete,
            "total_processed": total_processed or len(results),
            "last_selected_index": last_selected_index,
        }

        if metadata:
            data["metadata"] = metadata

        # Check if data has actually changed (avoid unnecessary I/O)
        data_hash = hash(json.dumps(data, sort_keys=True, default=str))
        if data_hash == self._last_save_hash:
            return True  # No changes, skip save

        try:
            # Atomic write: write to temp file, then rename
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                prefix=".session_recovery_",
                dir=self.path.parent
            )

            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)

                # Atomic rename (works on same filesystem)
                os.replace(temp_path, self.path)

                self._last_save_hash = data_hash
                self._dirty = False
                return True

            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        except Exception as e:
            print(f"[SessionRecovery] Error saving session: {e}")
            return False

    def has_recovery_file(self) -> bool:
        """Check if a recovery file exists.

        Returns:
            True if recovery file exists and is readable
        """
        return self.path.exists() and self.path.is_file()

    def load_recovery(self) -> Optional[Dict[str, Any]]:
        """Load recovery data from file.

        Returns:
            Recovery data dictionary, or None if load fails
        """
        if not self.has_recovery_file():
            return None

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate version
            version = data.get("version", 0)
            if version > self.VERSION:
                print(f"[SessionRecovery] Warning: Recovery file is from newer version ({version} > {self.VERSION})")

            return data

        except json.JSONDecodeError as e:
            print(f"[SessionRecovery] Error: Invalid JSON in recovery file: {e}")
            return None
        except Exception as e:
            print(f"[SessionRecovery] Error loading recovery file: {e}")
            return None

    def get_recovery_info(self) -> Optional[Dict[str, Any]]:
        """Get summary info from recovery file without loading full results.

        Returns:
            Dictionary with summary info:
            - timestamp: When the recovery was saved
            - input_directory: Source directory
            - result_count: Number of results
            - processing_complete: Whether processing finished
            - total_processed: Total bills processed
        """
        if not self.has_recovery_file():
            return None

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                "timestamp": data.get("timestamp"),
                "input_directory": data.get("input_directory", ""),
                "result_count": len(data.get("results", [])),
                "processing_complete": data.get("processing_complete", False),
                "total_processed": data.get("total_processed", 0),
                "version": data.get("version", 0),
            }

        except Exception as e:
            print(f"[SessionRecovery] Error reading recovery info: {e}")
            return None

    def clear_recovery(self) -> bool:
        """Delete the recovery file.

        Called after successful archive to prevent stale recovery.

        Returns:
            True if file was deleted or didn't exist, False on error
        """
        if not self.has_recovery_file():
            return True

        try:
            self.path.unlink()
            self._last_save_hash = None
            self._dirty = False
            return True
        except Exception as e:
            print(f"[SessionRecovery] Error deleting recovery file: {e}")
            return False

    def validate_recovery(self) -> Dict[str, Any]:
        """Validate a recovery file and check for issues.

        Returns:
            Dictionary with validation results:
            - valid: Whether the file is valid
            - issues: List of issues found
            - warnings: List of warnings
        """
        result = {
            "valid": True,
            "issues": [],
            "warnings": [],
        }

        if not self.has_recovery_file():
            result["valid"] = False
            result["issues"].append("Recovery file does not exist")
            return result

        data = self.load_recovery()
        if data is None:
            result["valid"] = False
            result["issues"].append("Failed to parse recovery file")
            return result

        # Check version
        version = data.get("version", 0)
        if version > self.VERSION:
            result["warnings"].append(f"Recovery file from newer version ({version})")

        # Check input directory exists
        input_dir = data.get("input_directory", "")
        if input_dir:
            input_path = Path(input_dir)
            if not input_path.exists():
                result["warnings"].append(f"Input directory no longer exists: {input_dir}")

        # Check result file paths
        results = data.get("results", [])
        missing_files = 0
        for r in results:
            front_file = r.get("front_file", "")
            if front_file and not Path(front_file).exists():
                missing_files += 1

        if missing_files > 0:
            result["warnings"].append(f"{missing_files} result files no longer exist")

        return result

    def mark_dirty(self):
        """Mark session as having unsaved changes."""
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        """Check if session has unsaved changes."""
        return self._dirty


# Singleton instance for easy access
_recovery_instance: Optional[SessionRecoveryManager] = None


def get_recovery_manager() -> SessionRecoveryManager:
    """Get the global recovery manager instance."""
    global _recovery_instance
    if _recovery_instance is None:
        _recovery_instance = SessionRecoveryManager()
    return _recovery_instance


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Session Recovery Manager CLI")
    parser.add_argument('command', choices=['info', 'validate', 'clear', 'test'])
    parser.add_argument('--file', '-f', help='Recovery file path')

    args = parser.parse_args()

    recovery = SessionRecoveryManager(Path(args.file) if args.file else None)

    if args.command == 'info':
        if recovery.has_recovery_file():
            info = recovery.get_recovery_info()
            print(f"Recovery file: {recovery.path}")
            print(f"Timestamp: {info.get('timestamp', 'N/A')}")
            print(f"Input directory: {info.get('input_directory', 'N/A')}")
            print(f"Results: {info.get('result_count', 0)}")
            print(f"Processing complete: {info.get('processing_complete', False)}")
            print(f"Total processed: {info.get('total_processed', 0)}")
        else:
            print("No recovery file found")

    elif args.command == 'validate':
        validation = recovery.validate_recovery()
        print(f"Valid: {validation['valid']}")
        if validation['issues']:
            print("Issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

    elif args.command == 'clear':
        if recovery.clear_recovery():
            print("Recovery file cleared")
        else:
            print("Failed to clear recovery file")

    elif args.command == 'test':
        # Test save/load cycle
        test_results = [
            {"position": 1, "serial": "A12345678B", "is_fancy": True},
            {"position": 2, "serial": "C87654321D", "is_fancy": False},
        ]

        print("Testing save...")
        if recovery.save_session(test_results, "/test/input", processing_complete=True):
            print("Save successful")
        else:
            print("Save failed")
            exit(1)

        print("\nTesting load...")
        data = recovery.load_recovery()
        if data:
            print(f"Loaded {len(data.get('results', []))} results")
            print(f"Input dir: {data.get('input_directory')}")
        else:
            print("Load failed")
            exit(1)

        print("\nTesting clear...")
        if recovery.clear_recovery():
            print("Clear successful")
        else:
            print("Clear failed")

        print("\nAll tests passed!")
