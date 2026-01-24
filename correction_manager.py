"""
Correction Manager - Phase 2
Manages manual corrections to OCR readings.

Features:
- Load/save corrections from YAML
- Apply corrections to processing results
- Track correction history
- Export/import for backup
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class Correction:
    """A single correction entry."""
    filename: str
    original_read: Optional[str]
    corrected_serial: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    corrected_by: str = "manual"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            'original_read': self.original_read,
            'corrected_serial': self.corrected_serial,
            'timestamp': self.timestamp,
            'corrected_by': self.corrected_by,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, filename: str, d: dict) -> 'Correction':
        return cls(
            filename=filename,
            original_read=d.get('original_read'),
            corrected_serial=d['corrected_serial'],
            timestamp=d.get('timestamp', datetime.now().isoformat()),
            corrected_by=d.get('corrected_by', 'manual'),
            notes=d.get('notes', '')
        )


class CorrectionManager:
    """
    Manages corrections for OCR misreads.

    Usage:
        cm = CorrectionManager()
        cm.add_correction("bill_0008.jpg", "G12345678A", "C12345678A")
        cm.save()

        # Later:
        serial = cm.get_corrected_serial("bill_0008.jpg", "G12345678A")
    """

    DEFAULT_PATH = Path(__file__).parent / "corrections.yaml"

    def __init__(self, path: Optional[Path] = None):
        self.path = path or self.DEFAULT_PATH
        self.corrections: Dict[str, Correction] = {}
        self._load()

    def _load(self):
        """Load corrections from YAML file."""
        if not self.path.exists():
            return

        with open(self.path, 'r') as f:
            data = yaml.safe_load(f) or {}

        corrections_data = data.get('corrections', {})
        for filename, corr_data in corrections_data.items():
            self.corrections[filename] = Correction.from_dict(filename, corr_data)

    def save(self):
        """Save corrections to YAML file."""
        data = {
            'version': '1.0',
            'updated': datetime.now().isoformat(),
            'count': len(self.corrections),
            'corrections': {
                filename: corr.to_dict()
                for filename, corr in self.corrections.items()
            }
        }

        with open(self.path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def add_correction(
        self,
        filename: str,
        original_read: Optional[str],
        corrected_serial: str,
        notes: str = "",
        corrected_by: str = "manual"
    ):
        """Add or update a correction."""
        self.corrections[filename] = Correction(
            filename=filename,
            original_read=original_read,
            corrected_serial=corrected_serial,
            notes=notes,
            corrected_by=corrected_by
        )

    def remove_correction(self, filename: str) -> bool:
        """Remove a correction. Returns True if found and removed."""
        if filename in self.corrections:
            del self.corrections[filename]
            return True
        return False

    def get_correction(self, filename: str) -> Optional[Correction]:
        """Get correction for a filename."""
        return self.corrections.get(filename)

    def get_corrected_serial(self, filename: str, original_read: Optional[str] = None) -> Optional[str]:
        """
        Get the corrected serial for a filename.
        Returns the corrected value if exists, otherwise None.
        """
        corr = self.corrections.get(filename)
        if corr:
            return corr.corrected_serial
        return None

    def apply_corrections(self, results: List[dict]) -> List[dict]:
        """
        Apply corrections to a list of processing results.
        Modifies in place and returns the list.
        """
        for result in results:
            filename = result.get('front_file', '')
            corr = self.get_correction(filename)
            if corr:
                result['original_serial'] = result.get('serial', '')
                result['serial'] = corr.corrected_serial
                result['corrected'] = True
        return results

    def has_correction(self, filename: str) -> bool:
        """Check if a correction exists for the filename."""
        return filename in self.corrections

    def get_all_corrections(self) -> Dict[str, Correction]:
        """Get all corrections."""
        return self.corrections.copy()

    def export_to_file(self, export_path: Path):
        """Export corrections to a backup file."""
        data = {
            'version': '1.0',
            'exported': datetime.now().isoformat(),
            'source': str(self.path),
            'count': len(self.corrections),
            'corrections': {
                filename: corr.to_dict()
                for filename, corr in self.corrections.items()
            }
        }
        with open(export_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def import_from_file(self, import_path: Path, overwrite: bool = False):
        """
        Import corrections from a backup file.
        If overwrite=False, only imports corrections that don't exist.
        """
        with open(import_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        corrections_data = data.get('corrections', {})
        imported = 0

        for filename, corr_data in corrections_data.items():
            if overwrite or filename not in self.corrections:
                self.corrections[filename] = Correction.from_dict(filename, corr_data)
                imported += 1

        return imported

    def get_statistics(self) -> dict:
        """Get statistics about corrections."""
        return {
            'total': len(self.corrections),
            'by_corrector': self._count_by_field('corrected_by'),
            'recent': self._get_recent_corrections(10)
        }

    def _count_by_field(self, field: str) -> dict:
        """Count corrections by a field value."""
        counts = {}
        for corr in self.corrections.values():
            value = getattr(corr, field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _get_recent_corrections(self, limit: int) -> List[dict]:
        """Get most recent corrections."""
        sorted_corrs = sorted(
            self.corrections.values(),
            key=lambda c: c.timestamp,
            reverse=True
        )[:limit]
        return [
            {'filename': c.filename, 'serial': c.corrected_serial, 'timestamp': c.timestamp}
            for c in sorted_corrs
        ]


# CLI for testing corrections
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Correction Manager CLI")
    parser.add_argument('command', choices=['list', 'add', 'remove', 'export', 'import', 'stats'])
    parser.add_argument('--file', '-f', help='Corrections file path')
    parser.add_argument('--filename', help='Bill filename for add/remove')
    parser.add_argument('--original', help='Original OCR read')
    parser.add_argument('--corrected', help='Corrected serial')
    parser.add_argument('--export-path', help='Path for export')
    parser.add_argument('--import-path', help='Path for import')

    args = parser.parse_args()

    cm = CorrectionManager(Path(args.file) if args.file else None)

    if args.command == 'list':
        print(f"Corrections file: {cm.path}")
        print(f"Total corrections: {len(cm.corrections)}")
        print()
        for filename, corr in cm.corrections.items():
            print(f"  {filename}:")
            print(f"    Original: {corr.original_read}")
            print(f"    Corrected: {corr.corrected_serial}")
            print(f"    Date: {corr.timestamp}")
            print()

    elif args.command == 'add':
        if not args.filename or not args.corrected:
            print("Error: --filename and --corrected required")
            exit(1)
        cm.add_correction(args.filename, args.original, args.corrected)
        cm.save()
        print(f"Added correction: {args.filename} -> {args.corrected}")

    elif args.command == 'remove':
        if not args.filename:
            print("Error: --filename required")
            exit(1)
        if cm.remove_correction(args.filename):
            cm.save()
            print(f"Removed correction for: {args.filename}")
        else:
            print(f"No correction found for: {args.filename}")

    elif args.command == 'export':
        if not args.export_path:
            print("Error: --export-path required")
            exit(1)
        cm.export_to_file(Path(args.export_path))
        print(f"Exported {len(cm.corrections)} corrections to {args.export_path}")

    elif args.command == 'import':
        if not args.import_path:
            print("Error: --import-path required")
            exit(1)
        imported = cm.import_from_file(Path(args.import_path))
        cm.save()
        print(f"Imported {imported} corrections from {args.import_path}")

    elif args.command == 'stats':
        stats = cm.get_statistics()
        print(f"Total corrections: {stats['total']}")
        print(f"By corrector: {stats['by_corrector']}")
        print("Recent corrections:")
        for item in stats['recent']:
            print(f"  {item['filename']}: {item['serial']} ({item['timestamp'][:10]})")
