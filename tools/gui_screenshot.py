"""Offscreen screenshot harness for the Dollar Detective GUI.

Launches the real MainWindow with no display (QT_QPA_PLATFORM=offscreen), lets
you load an existing strap and drive the view, then grabs the rendered window
(or a sub-widget) to a PNG. Used to *see* a GUI state after a change without a
human clicking -- logic still gets tested headlessly, this is the visual check.

Examples
--------
  # whole window, empty app, dark theme
  python tools/gui_screenshot.py --out /tmp/dd.png --theme dark

  # load strap 023, select the 4th bill, split view
  python tools/gui_screenshot.py --batch 023 --select 3 --view split_h --out /tmp/dd.png

  # just the preview pane
  python tools/gui_screenshot.py --batch 023 --select 0 --widget preview --out /tmp/prev.png

Notes
-----
- A loaded batch renders its preview from the CSV's cached alignment, so no YOLO
  model is loaded; the bill images must exist at the paths in the CSV.
- Views: front | back | stitched | split_v | split_h
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Screenshot the Dollar Detective GUI offscreen.")
    ap.add_argument("--out", default="/tmp/dd_gui.png", help="output PNG path")
    ap.add_argument("--batch", help="strap folder path, or a name under the straps dir (e.g. 023)")
    ap.add_argument("--select", type=int, default=0, help="result row to select (0-based)")
    ap.add_argument("--view", help="front|back|stitched|split_v|split_h")
    ap.add_argument("--theme", help="override theme: light|dark|system")
    ap.add_argument("--size", default="1500x950", help="window size WxH")
    ap.add_argument("--widget", choices=["preview", "results"], help="grab a sub-widget instead of the whole window")
    ap.add_argument("--settle", type=int, default=600, help="ms to let async rendering settle")
    ap.add_argument("--list", action="store_true", help="list available straps and exit")
    args = ap.parse_args()

    import faulthandler
    faulthandler.dump_traceback_later(45, exit=True)  # never hang silently offscreen

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QEventLoop, QTimer
    from gui.main_window import MainWindow

    # Startup modals block forever offscreen (and the update check hits the
    # network): the auto-update prompt and the session-recovery dialog. A
    # screenshot harness wants a clean, non-interactive launch, so stub both.
    MainWindow._start_update_check = lambda self, *a, **k: None
    MainWindow._check_for_recovery = lambda self, *a, **k: None
    from gui.theme_manager import apply_theme, get_combined_stylesheet
    from dollar_detective.settings_manager import get_settings
    from dollar_detective.resource_path import straps_dir

    def spin(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    app = QApplication(sys.argv)
    app.setApplicationName("Dollar Detective")
    app.setOrganizationName("DollarDetective")
    settings = get_settings()
    theme = args.theme or settings.ui.theme
    apply_theme(app, theme)
    app.setStyleSheet(get_combined_stylesheet(theme, settings.ui.font_size))

    if args.list:
        sd = Path(straps_dir())
        straps = sorted(p.name for p in sd.glob("*") if p.is_dir())
        print(f"straps dir: {sd}")
        print("  " + "  ".join(straps) if straps else "  (none)")
        os._exit(0)

    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
    except Exception:
        w, h = 1500, 950

    win = MainWindow()
    win.resize(w, h)
    win.show()
    spin(args.settle)

    if args.batch:
        bp = Path(args.batch)
        if not bp.exists():
            bp = Path(straps_dir()) / args.batch
        if not bp.exists():
            print(f"batch not found: {args.batch}")
        else:
            try:
                win.results_list.refresh_batch_list()
                win.results_list._load_batch(bp)
                win.results_list._current_batch_path = bp
                spin(300)
                tree = win.results_list.tree
                n = tree.topLevelItemCount()
                if n:
                    i = max(0, min(args.select, n - 1))
                    tree.setCurrentItem(tree.topLevelItem(i))
                print(f"loaded {bp.name}: {n} rows, selected row {min(args.select, max(0, n-1))}")
            except Exception as e:
                print(f"batch load failed: {e}")
            spin(args.settle)

    if args.view:
        try:
            win.preview_panel._on_view_mode_clicked(args.view)
        except Exception as e:
            print(f"view set failed: {e}")
        spin(args.settle)

    target = win
    if args.widget == "preview":
        target = win.preview_panel
    elif args.widget == "results":
        target = win.results_list
    pix = target.grab()
    ok = pix.save(args.out)
    print(f"saved {args.out}: ok={ok} size={pix.width()}x{pix.height()}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
