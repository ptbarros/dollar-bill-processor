"""
Theme Manager - Dark/Light theme support for the Dollar Detective.

Provides cross-platform theming using Qt's Fusion style with custom palettes.
"""

import os
import sys
import tempfile
from PySide6.QtWidgets import QApplication, QStyleFactory
from PySide6.QtGui import QPalette, QColor, QPixmap, QPainter, QBrush
from PySide6.QtCore import Qt, QRectF


def get_system_is_dark() -> bool:
    """Detect if the system is using dark mode.

    Returns True if dark mode is detected, False otherwise.
    """
    # On Windows, check the registry for dark mode setting
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            return value == 0  # 0 = dark mode, 1 = light mode
        except (WindowsError, FileNotFoundError, OSError):
            return False

    # On macOS, check system appearance
    elif sys.platform == "darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True
            )
            return result.stdout.strip().lower() == "dark"
        except Exception:
            return False

    # On Linux, try to detect from environment or desktop settings
    else:
        # Check common environment variables
        import os

        # GTK theme hint
        gtk_theme = os.environ.get("GTK_THEME", "").lower()
        if "dark" in gtk_theme:
            return True

        # Qt platform theme
        qt_theme = os.environ.get("QT_QPA_PLATFORMTHEME", "").lower()

        # Try to read GNOME/GTK settings
        try:
            import subprocess
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                capture_output=True, text=True
            )
            if "dark" in result.stdout.lower():
                return True
        except Exception:
            pass

        # Try KDE settings
        try:
            from pathlib import Path
            kde_config = Path.home() / ".config" / "kdeglobals"
            if kde_config.exists():
                content = kde_config.read_text()
                # Very basic check - KDE dark themes often have dark in the name
                if "dark" in content.lower():
                    return True
        except Exception:
            pass

        return False


def create_dark_palette() -> QPalette:
    """Create a dark color palette for the application."""
    palette = QPalette()

    # Base colors
    dark_gray = QColor(45, 45, 45)
    darker_gray = QColor(35, 35, 35)
    light_gray = QColor(60, 60, 60)
    text_color = QColor(220, 220, 220)
    disabled_text = QColor(127, 127, 127)
    highlight = QColor(42, 130, 218)
    highlight_text = QColor(255, 255, 255)
    link_color = QColor(42, 130, 218)

    # Window and base
    palette.setColor(QPalette.Window, dark_gray)
    palette.setColor(QPalette.WindowText, text_color)
    palette.setColor(QPalette.Base, darker_gray)
    palette.setColor(QPalette.AlternateBase, dark_gray)

    # Text
    palette.setColor(QPalette.Text, text_color)
    palette.setColor(QPalette.BrightText, Qt.white)
    palette.setColor(QPalette.PlaceholderText, disabled_text)

    # Buttons
    palette.setColor(QPalette.Button, dark_gray)
    palette.setColor(QPalette.ButtonText, text_color)

    # Selections
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, highlight_text)

    # Links
    palette.setColor(QPalette.Link, link_color)
    palette.setColor(QPalette.LinkVisited, QColor(150, 100, 200))

    # Tooltips
    palette.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipText, text_color)

    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text)

    # Misc
    palette.setColor(QPalette.Light, light_gray)
    palette.setColor(QPalette.Midlight, QColor(55, 55, 55))
    palette.setColor(QPalette.Dark, QColor(25, 25, 25))
    palette.setColor(QPalette.Mid, QColor(40, 40, 40))
    palette.setColor(QPalette.Shadow, QColor(20, 20, 20))

    return palette


def create_light_palette() -> QPalette:
    """Create a light color palette for the application."""
    palette = QPalette()

    # Base colors - standard light theme
    white = QColor(255, 255, 255)
    light_gray = QColor(240, 240, 240)
    mid_gray = QColor(200, 200, 200)
    text_color = QColor(0, 0, 0)
    disabled_text = QColor(127, 127, 127)
    highlight = QColor(42, 130, 218)
    highlight_text = QColor(255, 255, 255)

    # Window and base
    palette.setColor(QPalette.Window, light_gray)
    palette.setColor(QPalette.WindowText, text_color)
    palette.setColor(QPalette.Base, white)
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))

    # Text
    palette.setColor(QPalette.Text, text_color)
    palette.setColor(QPalette.BrightText, Qt.white)
    palette.setColor(QPalette.PlaceholderText, disabled_text)

    # Buttons
    palette.setColor(QPalette.Button, light_gray)
    palette.setColor(QPalette.ButtonText, text_color)

    # Selections
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, highlight_text)

    # Links
    palette.setColor(QPalette.Link, QColor(0, 0, 255))
    palette.setColor(QPalette.LinkVisited, QColor(128, 0, 128))

    # Tooltips
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, text_color)

    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text)

    # Misc
    palette.setColor(QPalette.Light, white)
    palette.setColor(QPalette.Midlight, QColor(227, 227, 227))
    palette.setColor(QPalette.Dark, QColor(160, 160, 160))
    palette.setColor(QPalette.Mid, mid_gray)
    palette.setColor(QPalette.Shadow, QColor(105, 105, 105))

    return palette


def is_dark_theme(theme: str) -> bool:
    """Check if the given theme setting results in dark mode.

    Args:
        theme: One of "system", "light", or "dark"

    Returns:
        True if dark mode should be used
    """
    if theme == "system":
        return get_system_is_dark()
    elif theme == "dark":
        return True
    else:
        return False


def get_dark_stylesheet() -> str:
    """Get the stylesheet additions for dark mode.

    Returns CSS rules that help widgets that don't fully respect the palette.
    """
    return """
            QToolTip {
                background-color: #323232;
                color: #dcdcdc;
                border: 1px solid #555;
                padding: 2px;
            }
            QMenu {
                background-color: #2d2d2d;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background-color: #2a82da;
            }
            QMenuBar::item:selected {
                background-color: #2a82da;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                selection-background-color: #2a82da;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 14px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                min-height: 20px;
                border-radius: 3px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar:horizontal {
                background-color: #2d2d2d;
                height: 14px;
            }
            QScrollBar::handle:horizontal {
                background-color: #555;
                min-width: 20px;
                border-radius: 3px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #666;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
                border: none;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background: none;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                color: #dcdcdc;
                padding: 4px;
                border: 1px solid #555;
            }
            QTableView {
                gridline-color: #555;
            }
            QTreeView::item:hover {
                background-color: #3c3c3c;
            }
            QGroupBox {
                border: 1px solid #555;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #232323;
                border: 1px solid #555;
                padding: 3px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #2a82da;
            }
            QTextEdit, QPlainTextEdit {
                background-color: #232323;
                border: 1px solid #555;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a82da;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666;
            }
            QTabWidget::pane {
                border: 1px solid #555;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 6px 12px;
            }
            QTabBar::tab:selected {
                background-color: #3c3c3c;
                border-bottom-color: #3c3c3c;
            }
            QTabBar::tab:hover:!selected {
                background-color: #353535;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #232323;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
            }
            QSlider::groove:horizontal {
                background-color: #232323;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #2a82da;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {
                border: 1px solid #555;
                background-color: #232323;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2a82da;
                background-color: #2a82da;
            }
            QSplitter::handle {
                background-color: #3c3c3c;
            }
            QSplitter::handle:hover {
                background-color: #2a82da;
            }
        """


def apply_theme(app: QApplication, theme: str = "system") -> None:
    """Apply a theme to the application.

    Args:
        app: The QApplication instance
        theme: One of "system", "light", or "dark"
    """
    # Use Fusion style for consistent cross-platform appearance
    app.setStyle(QStyleFactory.create("Fusion"))

    # Determine which palette to use
    use_dark = is_dark_theme(theme)

    # Apply the appropriate palette
    if use_dark:
        palette = create_dark_palette()
    else:
        palette = create_light_palette()

    app.setPalette(palette)

    # Store the theme state for stylesheet generation
    # The actual stylesheet is applied by combine_with_font_stylesheet
    app.setProperty("_dark_theme", use_dark)


_grip_cache = {}


def _splitter_grip_path(orientation: str, color_hex: str):
    """Path to a small rounded 'grip' pixmap centred on a splitter handle: a short
    vertical capsule for a horizontal handle (vertical divider), a horizontal one
    for a vertical handle. Generated + cached on first use; returns None on any
    failure so the caller can fall back to a plain line."""
    key = (orientation, color_hex)
    cached = _grip_cache.get(key)
    if cached and os.path.exists(cached):
        return cached
    try:
        # Long axis runs ALONG the divider so the middle reads as a thicker grip.
        if orientation == "horizontal":     # vertical divider -> vertical capsule
            w, h = 5, 26
        else:                               # horizontal divider -> horizontal capsule
            w, h = 26, 5
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(color_hex)))
        r = min(w, h) / 2.0
        p.drawRoundedRect(QRectF(0, 0, w, h), r, r)
        p.end()
        d = os.path.join(tempfile.gettempdir(), "dollardetective_theme")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"grip_{orientation}_{color_hex.lstrip('#')}.png")
        if pm.save(path, "PNG"):
            forward = path.replace("\\", "/")
            _grip_cache[key] = forward
            return forward
    except Exception:
        pass
    return None


def get_combined_stylesheet(theme: str, font_size: int) -> str:
    """Get a combined stylesheet with theme colors and font sizes.

    Args:
        theme: One of "system", "light", or "dark"
        font_size: Base font size in points

    Returns:
        Combined CSS stylesheet string
    """
    use_dark = is_dark_theme(theme)

    # Start with font size rules
    stylesheet = f"""
        QWidget {{
            font-size: {font_size}pt;
        }}
        QTreeWidget {{
            font-size: {font_size}pt;
        }}
        QTreeWidget::item {{
            padding: {max(2, font_size // 4)}px;
        }}
        QListWidget {{
            font-size: {font_size}pt;
        }}
        QTableWidget {{
            font-size: {font_size}pt;
        }}
        QPushButton {{
            font-size: {font_size}pt;
        }}
        QLabel {{
            font-size: {font_size}pt;
        }}
        QLineEdit {{
            font-size: {font_size}pt;
        }}
        QComboBox {{
            font-size: {font_size}pt;
        }}
        QSpinBox, QDoubleSpinBox {{
            font-size: {font_size}pt;
        }}
        QGroupBox {{
            font-size: {font_size}pt;
        }}
        QGroupBox::title {{
            font-size: {font_size}pt;
        }}
        QTabWidget::tab-bar {{
            font-size: {font_size}pt;
        }}
        QTabBar::tab {{
            font-size: {font_size}pt;
        }}
        QMenuBar {{
            font-size: {font_size}pt;
        }}
        QMenu {{
            font-size: {font_size}pt;
        }}
        QStatusBar {{
            font-size: {font_size}pt;
        }}
    """

    # Add dark mode styles if needed
    if use_dark:
        stylesheet += get_dark_stylesheet()

    # Resize divider (splitter handle): a THIN centre line with a small rounded
    # grip in the middle, rather than a thick bar -- easy to see and grab without
    # looking heavy. The handle keeps a comfortable hit width but paints only a
    # ~1.5px line (via a gradient that's transparent except a central band) plus a
    # centred capsule image. Appended last so it overrides the fainter dark rule.
    handle_line = "#8a8a8a" if use_dark else "#8f8f8f"
    grip_color = "#9a9a9a" if use_dark else "#7d7d7d"
    hover = "#2a82da"
    gh, gv = _splitter_grip_path("horizontal", grip_color), _splitter_grip_path("vertical", grip_color)
    ghh, gvh = _splitter_grip_path("horizontal", hover), _splitter_grip_path("vertical", hover)

    def _img(path):
        return f'image: url("{path}");' if path else ""

    stylesheet += f"""
        QSplitter::handle {{
            background-color: transparent;
        }}
        QSplitter::handle:horizontal {{
            width: 8px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent, stop:0.40 transparent,
                stop:0.41 {handle_line}, stop:0.59 {handle_line},
                stop:0.60 transparent, stop:1 transparent);
            {_img(gh)}
        }}
        QSplitter::handle:vertical {{
            height: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 transparent, stop:0.40 transparent,
                stop:0.41 {handle_line}, stop:0.59 {handle_line},
                stop:0.60 transparent, stop:1 transparent);
            {_img(gv)}
        }}
        QSplitter::handle:horizontal:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent, stop:0.37 transparent,
                stop:0.38 {hover}, stop:0.62 {hover},
                stop:0.63 transparent, stop:1 transparent);
            {_img(ghh)}
        }}
        QSplitter::handle:vertical:hover {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 transparent, stop:0.37 transparent,
                stop:0.38 {hover}, stop:0.62 {hover},
                stop:0.63 transparent, stop:1 transparent);
            {_img(gvh)}
        }}
    """

    return stylesheet
