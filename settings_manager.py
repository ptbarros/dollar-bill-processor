"""
Settings Manager - Phase 2
Persistent user settings for the Dollar Bill Processor.

Features:
- Pattern enable/disable states
- Confidence thresholds
- UI preferences
- Last used directories
- Export settings
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class ProcessingSettings:
    """Processing-related settings."""
    confidence_threshold: float = 0.5
    use_gpu: bool = False
    verify_pairs: bool = True
    jpeg_quality: int = 95
    multi_pass_detection: bool = True
    max_detection_passes: int = 5
    crop_all: bool = False  # Crop all bills, not just fancy ones


@dataclass
class UISettings:
    """UI-related settings."""
    last_input_dir: str = ""
    last_output_dir: str = ""
    window_width: int = 1200
    window_height: int = 800
    window_x: int = 100
    window_y: int = 100
    results_sort_column: str = "position"
    results_sort_ascending: bool = True
    show_thumbnails: bool = True
    thumbnail_size: int = 200
    theme: str = "system"  # system, light, dark
    font_size: int = 10  # Base font size in points (default 10)
    default_fancy_color: str = "#2e7d32"  # Default green for fancy bills (user-customizable)


@dataclass
class ExportSettings:
    """Export-related settings."""
    default_format: str = "csv"  # csv, excel, html
    include_thumbnails: bool = True
    excel_template: str = ""
    html_template: str = ""
    auto_export_csv: bool = True  # Auto-generate CSV after processing
    auto_export_summary: bool = True  # Auto-generate summary after processing


@dataclass
class CropSettings:
    """Crop region settings (percentages 0.0-1.0)."""
    # Front seal crop
    front_seal_x: float = 0.605
    front_seal_y: float = 0.233
    front_seal_w: float = 0.254
    front_seal_h: float = 0.537
    # Back seal crop
    back_seal_x: float = 0.635
    back_seal_y: float = 0.221
    back_seal_w: float = 0.261
    back_seal_h: float = 0.557


@dataclass
class MonitorSettings:
    """Monitor mode settings for real-time processing."""
    watch_directory: str = ""  # Directory to watch for new files
    output_directory: str = ""  # Output directory for fancy bill crops
    archive_directory: str = ""  # Where to move processed batches
    poll_interval: float = 2.0  # Seconds between directory checks
    file_settle_time: float = 1.0  # Wait for file write completion
    auto_archive: bool = True  # Move files to timestamped dir on stop


class SettingsManager:
    """
    Manages persistent user settings.

    Usage:
        settings = SettingsManager()
        settings.processing.use_gpu = True
        settings.ui.last_input_dir = "/path/to/scans"
        settings.save()
    """

    DEFAULT_PATH = Path(__file__).parent / "user_settings.yaml"

    def __init__(self, path: Optional[Path] = None):
        self.path = path or self.DEFAULT_PATH
        self.processing = ProcessingSettings()
        self.ui = UISettings()
        self.export = ExportSettings()
        self.crop = CropSettings()
        self.monitor = MonitorSettings()
        self.pattern_states: Dict[str, bool] = {}  # Pattern name -> enabled
        self.pattern_colors: Dict[str, str] = {}  # Pattern name -> hex color
        self.custom_values: Dict[str, Any] = {}  # Arbitrary user values
        self._load()

    def _load(self):
        """Load settings from YAML file."""
        if not self.path.exists():
            return

        with open(self.path, 'r') as f:
            data = yaml.safe_load(f) or {}

        # Load processing settings
        if 'processing' in data:
            proc = data['processing']
            self.processing.confidence_threshold = proc.get('confidence_threshold', 0.5)
            self.processing.use_gpu = proc.get('use_gpu', False)
            self.processing.verify_pairs = proc.get('verify_pairs', True)
            self.processing.jpeg_quality = proc.get('jpeg_quality', 95)
            self.processing.multi_pass_detection = proc.get('multi_pass_detection', True)
            self.processing.max_detection_passes = proc.get('max_detection_passes', 5)
            self.processing.crop_all = proc.get('crop_all', False)

        # Load UI settings
        if 'ui' in data:
            ui = data['ui']
            self.ui.last_input_dir = ui.get('last_input_dir', "")
            self.ui.last_output_dir = ui.get('last_output_dir', "")
            self.ui.window_width = ui.get('window_width', 1200)
            self.ui.window_height = ui.get('window_height', 800)
            self.ui.window_x = ui.get('window_x', 100)
            self.ui.window_y = ui.get('window_y', 100)
            self.ui.results_sort_column = ui.get('results_sort_column', 'position')
            self.ui.results_sort_ascending = ui.get('results_sort_ascending', True)
            self.ui.show_thumbnails = ui.get('show_thumbnails', True)
            self.ui.thumbnail_size = ui.get('thumbnail_size', 200)
            self.ui.theme = ui.get('theme', 'system')
            self.ui.font_size = ui.get('font_size', 10)
            self.ui.default_fancy_color = ui.get('default_fancy_color', '#2e7d32')

        # Load export settings
        if 'export' in data:
            exp = data['export']
            self.export.default_format = exp.get('default_format', 'csv')
            self.export.include_thumbnails = exp.get('include_thumbnails', True)
            self.export.excel_template = exp.get('excel_template', '')
            self.export.html_template = exp.get('html_template', '')
            self.export.auto_export_csv = exp.get('auto_export_csv', True)
            self.export.auto_export_summary = exp.get('auto_export_summary', True)

        # Load crop settings
        if 'crop' in data:
            crop = data['crop']
            self.crop.front_seal_x = crop.get('front_seal_x', 0.605)
            self.crop.front_seal_y = crop.get('front_seal_y', 0.233)
            self.crop.front_seal_w = crop.get('front_seal_w', 0.254)
            self.crop.front_seal_h = crop.get('front_seal_h', 0.537)
            self.crop.back_seal_x = crop.get('back_seal_x', 0.635)
            self.crop.back_seal_y = crop.get('back_seal_y', 0.221)
            self.crop.back_seal_w = crop.get('back_seal_w', 0.261)
            self.crop.back_seal_h = crop.get('back_seal_h', 0.557)

        # Load monitor settings
        if 'monitor' in data:
            mon = data['monitor']
            self.monitor.watch_directory = mon.get('watch_directory', '')
            self.monitor.output_directory = mon.get('output_directory', '')
            self.monitor.archive_directory = mon.get('archive_directory', '')
            self.monitor.poll_interval = mon.get('poll_interval', 2.0)
            self.monitor.file_settle_time = mon.get('file_settle_time', 1.0)
            self.monitor.auto_archive = mon.get('auto_archive', True)

        # Load pattern states
        self.pattern_states = data.get('pattern_states', {})

        # Load pattern colors
        self.pattern_colors = data.get('pattern_colors', {})

        # Load custom values
        self.custom_values = data.get('custom_values', {})

    def save(self):
        """Save settings to YAML file."""
        data = {
            'version': '1.0',
            'updated': datetime.now().isoformat(),
            'processing': {
                'confidence_threshold': self.processing.confidence_threshold,
                'use_gpu': self.processing.use_gpu,
                'verify_pairs': self.processing.verify_pairs,
                'jpeg_quality': self.processing.jpeg_quality,
                'multi_pass_detection': self.processing.multi_pass_detection,
                'max_detection_passes': self.processing.max_detection_passes,
                'crop_all': self.processing.crop_all,
            },
            'ui': {
                'last_input_dir': self.ui.last_input_dir,
                'last_output_dir': self.ui.last_output_dir,
                'window_width': self.ui.window_width,
                'window_height': self.ui.window_height,
                'window_x': self.ui.window_x,
                'window_y': self.ui.window_y,
                'results_sort_column': self.ui.results_sort_column,
                'results_sort_ascending': self.ui.results_sort_ascending,
                'show_thumbnails': self.ui.show_thumbnails,
                'thumbnail_size': self.ui.thumbnail_size,
                'theme': self.ui.theme,
                'font_size': self.ui.font_size,
                'default_fancy_color': self.ui.default_fancy_color,
            },
            'export': {
                'default_format': self.export.default_format,
                'include_thumbnails': self.export.include_thumbnails,
                'excel_template': self.export.excel_template,
                'html_template': self.export.html_template,
                'auto_export_csv': self.export.auto_export_csv,
                'auto_export_summary': self.export.auto_export_summary,
            },
            'crop': {
                'front_seal_x': self.crop.front_seal_x,
                'front_seal_y': self.crop.front_seal_y,
                'front_seal_w': self.crop.front_seal_w,
                'front_seal_h': self.crop.front_seal_h,
                'back_seal_x': self.crop.back_seal_x,
                'back_seal_y': self.crop.back_seal_y,
                'back_seal_w': self.crop.back_seal_w,
                'back_seal_h': self.crop.back_seal_h,
            },
            'monitor': {
                'watch_directory': self.monitor.watch_directory,
                'output_directory': self.monitor.output_directory,
                'archive_directory': self.monitor.archive_directory,
                'poll_interval': self.monitor.poll_interval,
                'file_settle_time': self.monitor.file_settle_time,
                'auto_archive': self.monitor.auto_archive,
            },
            'pattern_states': self.pattern_states,
            'pattern_colors': self.pattern_colors,
            'custom_values': self.custom_values,
        }

        with open(self.path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_pattern_enabled(self, pattern_name: str, default: bool = True) -> bool:
        """Get whether a pattern is enabled."""
        return self.pattern_states.get(pattern_name, default)

    def set_pattern_enabled(self, pattern_name: str, enabled: bool):
        """Set whether a pattern is enabled."""
        self.pattern_states[pattern_name] = enabled

    def get_enabled_patterns(self) -> List[str]:
        """Get list of explicitly enabled patterns."""
        return [name for name, enabled in self.pattern_states.items() if enabled]

    def get_disabled_patterns(self) -> List[str]:
        """Get list of explicitly disabled patterns."""
        return [name for name, enabled in self.pattern_states.items() if not enabled]

    def get_pattern_color(self, pattern_name: str, default: str = "") -> str:
        """Get custom color for a pattern (hex format like '#FF0000')."""
        return self.pattern_colors.get(pattern_name, default)

    def set_pattern_color(self, pattern_name: str, color: str):
        """Set custom color for a pattern (hex format like '#FF0000')."""
        if color:
            self.pattern_colors[pattern_name] = color
        elif pattern_name in self.pattern_colors:
            del self.pattern_colors[pattern_name]

    def set_custom_value(self, key: str, value: Any):
        """Set a custom user value."""
        self.custom_values[key] = value

    def get_custom_value(self, key: str, default: Any = None) -> Any:
        """Get a custom user value."""
        return self.custom_values.get(key, default)

    def update_window_geometry(self, x: int, y: int, width: int, height: int):
        """Update window position and size."""
        self.ui.window_x = x
        self.ui.window_y = y
        self.ui.window_width = width
        self.ui.window_height = height

    def get_window_geometry(self) -> tuple:
        """Get window geometry as (x, y, width, height)."""
        return (self.ui.window_x, self.ui.window_y,
                self.ui.window_width, self.ui.window_height)

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.processing = ProcessingSettings()
        self.ui = UISettings()
        self.export = ExportSettings()
        self.crop = CropSettings()
        self.monitor = MonitorSettings()
        self.pattern_states = {}
        self.custom_values = {}

    def export_settings(self, export_path: Path):
        """Export settings to a backup file."""
        # Save first to ensure current state
        self.save()
        # Copy the file
        import shutil
        shutil.copy(self.path, export_path)

    def import_settings(self, import_path: Path):
        """Import settings from a backup file."""
        import shutil
        shutil.copy(import_path, self.path)
        self._load()


# Singleton instance for easy access
_settings_instance: Optional[SettingsManager] = None


def get_settings() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SettingsManager()
    return _settings_instance


# CLI for testing settings
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Settings Manager CLI")
    parser.add_argument('command', choices=['show', 'set', 'reset', 'export', 'import'])
    parser.add_argument('--file', '-f', help='Settings file path')
    parser.add_argument('--key', help='Setting key (e.g., processing.use_gpu)')
    parser.add_argument('--value', help='Setting value')
    parser.add_argument('--export-path', help='Path for export')
    parser.add_argument('--import-path', help='Path for import')

    args = parser.parse_args()

    settings = SettingsManager(Path(args.file) if args.file else None)

    if args.command == 'show':
        print(f"Settings file: {settings.path}")
        print()
        print("Processing:")
        print(f"  Confidence threshold: {settings.processing.confidence_threshold}")
        print(f"  Use GPU: {settings.processing.use_gpu}")
        print(f"  Verify pairs: {settings.processing.verify_pairs}")
        print(f"  JPEG quality: {settings.processing.jpeg_quality}")
        print(f"  Multi-pass detection: {settings.processing.multi_pass_detection}")
        print()
        print("UI:")
        print(f"  Last input dir: {settings.ui.last_input_dir}")
        print(f"  Last output dir: {settings.ui.last_output_dir}")
        print(f"  Theme: {settings.ui.theme}")
        print(f"  Window: {settings.ui.window_width}x{settings.ui.window_height}")
        print()
        print("Export:")
        print(f"  Default format: {settings.export.default_format}")
        print(f"  Include thumbnails: {settings.export.include_thumbnails}")
        print()
        print(f"Pattern states: {len(settings.pattern_states)} customized")
        print(f"Custom values: {len(settings.custom_values)} stored")

    elif args.command == 'set':
        if not args.key or args.value is None:
            print("Error: --key and --value required")
            exit(1)

        # Parse the key path
        parts = args.key.split('.')
        if len(parts) == 2:
            section, key = parts
            if section == 'processing':
                obj = settings.processing
            elif section == 'ui':
                obj = settings.ui
            elif section == 'export':
                obj = settings.export
            else:
                print(f"Unknown section: {section}")
                exit(1)

            # Try to set the value with appropriate type
            current = getattr(obj, key, None)
            if isinstance(current, bool):
                value = args.value.lower() in ('true', '1', 'yes')
            elif isinstance(current, int):
                value = int(args.value)
            elif isinstance(current, float):
                value = float(args.value)
            else:
                value = args.value

            setattr(obj, key, value)
            settings.save()
            print(f"Set {args.key} = {value}")
        else:
            print("Error: key must be in format section.key (e.g., processing.use_gpu)")

    elif args.command == 'reset':
        settings.reset_to_defaults()
        settings.save()
        print("Settings reset to defaults")

    elif args.command == 'export':
        if not args.export_path:
            print("Error: --export-path required")
            exit(1)
        settings.export_settings(Path(args.export_path))
        print(f"Settings exported to {args.export_path}")

    elif args.command == 'import':
        if not args.import_path:
            print("Error: --import-path required")
            exit(1)
        settings.import_settings(Path(args.import_path))
        print(f"Settings imported from {args.import_path}")
