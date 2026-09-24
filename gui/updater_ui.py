"""GUI glue for the in-app updater (updater.py).

Runs the network check off the UI thread, prompts when a newer stable release
exists, and either downloads+launches the matching Windows installer or opens the
release page (Linux/macOS, or when no matching asset is found).
"""

import sys
import webbrowser

from PySide6.QtCore import QThread, Signal, Qt, QEventLoop
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QApplication

import updater


class UpdateCheckThread(QThread):
    """Background GitHub Releases check; emits UpdateInfo or None."""
    finished_check = Signal(object)

    def run(self):
        self.finished_check.emit(updater.check_for_update())


class _DownloadThread(QThread):
    progress = Signal(int, int)   # downloaded, total
    done = Signal(str)            # local path, or "" on failure

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self.url = url

    def run(self):
        path = updater.download_asset(self.url, lambda d, t: self.progress.emit(d, t))
        self.done.emit(path or "")


def check_in_background(parent, on_result):
    """Start a background update check; on_result(info_or_none) runs on the UI
    thread. Returns the thread so the caller can keep a reference (avoid GC)."""
    t = UpdateCheckThread(parent)
    t.finished_check.connect(on_result)
    t.start()
    return t


def prompt_and_apply(parent, info):
    """Show the update prompt for `info` (updater.UpdateInfo). On accept: on
    Windows download the matching installer and launch it (then quit the app);
    elsewhere open the release page."""
    asset = info.asset_for_current_edition()
    can_autoinstall = (sys.platform == "win32" and asset
                       and asset["name"].endswith(".exe"))

    from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                   QTextBrowser, QPushButton)

    dlg = QDialog(parent)
    dlg.setWindowTitle("Update available")
    dlg.setMinimumWidth(480)
    v = QVBoxLayout(dlg)

    header = QLabel(f"<b>Dollar Detective {info.latest_version}</b> is available "
                    f"(you have {info.current_version}).")
    header.setWordWrap(True)
    v.addWidget(header)

    # "What's new" -- the release's notes (GitHub auto-generated "What's Changed").
    if info.notes:
        v.addWidget(QLabel("What's new:"))
        notes = QTextBrowser()
        notes.setOpenExternalLinks(True)
        try:
            notes.setMarkdown(info.notes)
        except Exception:
            notes.setPlainText(info.notes)
        notes.setMinimumHeight(200)
        v.addWidget(notes, 1)

    ask = QLabel("Download and install it now? The app will close to finish installing."
                 if can_autoinstall else "Open the download page in your browser?")
    ask.setWordWrap(True)
    v.addWidget(ask)

    row = QHBoxLayout()
    row.addStretch()
    later_btn = QPushButton("Later")
    accept_btn = QPushButton("Download && Install" if can_autoinstall else "Open Download Page")
    accept_btn.setDefault(True)
    row.addWidget(later_btn)
    row.addWidget(accept_btn)
    v.addLayout(row)

    accepted = {"v": False}
    later_btn.clicked.connect(dlg.reject)
    accept_btn.clicked.connect(lambda: (accepted.__setitem__("v", True), dlg.accept()))
    dlg.exec()
    if not accepted["v"]:
        return

    if not can_autoinstall:
        webbrowser.open(info.release_url)
        return

    # Windows: download + launch the installer. Record the version we're leaving
    # first, so it can be one-click reverted to afterwards.
    from settings_manager import get_settings
    _download_and_launch(
        parent, asset, info.release_url,
        on_before_launch=lambda: get_settings().set_previous_version(info.current_version),
    )


def _download_and_launch(parent, asset, release_url, on_before_launch=None):
    """Download `asset` ({'name','url'}) with a progress dialog, then launch the
    Windows installer and quit the app. Falls back to opening `release_url` on
    failure. `on_before_launch()` runs once the download succeeds, just before the
    installer is launched (used to record/clear the revert target)."""
    prog = QProgressDialog("Downloading…", "Cancel", 0, 100, parent)
    prog.setWindowTitle("Downloading")
    prog.setWindowModality(Qt.WindowModal)
    prog.setMinimumDuration(0)
    prog.setAutoClose(False)
    prog.setAutoReset(False)

    state = {"path": "", "cancelled": False}
    loop = QEventLoop()
    t = _DownloadThread(asset["url"], parent)
    t.progress.connect(lambda d, total: (prog.setMaximum(total or 0), prog.setValue(d)))
    t.done.connect(lambda p: (state.__setitem__("path", p), loop.quit()))
    prog.canceled.connect(lambda: (state.__setitem__("cancelled", True), loop.quit()))

    t.start()
    prog.show()
    loop.exec()

    if state["cancelled"]:
        prog.reset()
        return
    t.wait()

    path = state["path"]
    if not path:
        prog.reset()
        QMessageBox.warning(parent, "Download failed",
                            "Couldn't download it. Opening the download page instead.")
        webbrowser.open(release_url)
        return

    # Download finished. The installer can take several seconds to self-extract
    # and show its own window; leaving the bar full/idle makes it look frozen.
    # Switch to an indeterminate "Launching installer…" state so the hand-off is
    # visibly in progress until the app quits and the installer takes over.
    prog.setCancelButton(None)          # can't cancel once we're launching
    prog.setLabelText("Launching installer…")
    prog.setRange(0, 0)                 # busy/marquee indicator
    QApplication.processEvents()

    if on_before_launch:
        try:
            on_before_launch()
        except Exception:
            pass

    if updater.launch_installer_and_exit(path):
        QApplication.quit()
    else:
        prog.reset()
        webbrowser.open(release_url)


class _ReleaseLookupThread(QThread):
    """Background lookup of a past release's installer asset by version."""
    found = Signal(object)   # (release_url, asset_or_None)

    def __init__(self, version, parent=None):
        super().__init__(parent)
        self.version = version

    def run(self):
        self.found.emit(updater.release_asset_for_version(self.version))


def revert_to_previous(parent, version):
    """One-click revert to `version` (the one recorded before the last update).
    Looks up that release's installer for the running edition and, on Windows,
    downloads + launches it; otherwise opens the release page."""
    if not version:
        return
    if QMessageBox.question(
            parent, "Revert update",
            f"Reinstall Dollar Detective {version}?\n\n"
            "The app will close to finish reinstalling. Your patterns, settings and "
            "saved work are kept.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
        return

    # Find the release's installer (network) behind a small busy dialog.
    busy = QProgressDialog(f"Finding version {version}…", None, 0, 0, parent)
    busy.setWindowTitle("Revert update")
    busy.setWindowModality(Qt.WindowModal)
    busy.setMinimumDuration(0)

    result = {"url": updater.RELEASES_PAGE, "asset": None}
    loop = QEventLoop()
    lookup = _ReleaseLookupThread(version, parent)
    lookup.found.connect(lambda r: (result.__setitem__("url", r[0]),
                                    result.__setitem__("asset", r[1]),
                                    loop.quit()))
    lookup.start()
    busy.show()
    loop.exec()
    busy.reset()
    lookup.wait()

    asset = result["asset"]
    release_url = result["url"]
    can_autoinstall = (sys.platform == "win32" and asset
                       and asset["name"].endswith(".exe"))
    if not can_autoinstall:
        # No matching installer (non-Windows, or asset missing) -> open the page.
        webbrowser.open(release_url)
        return

    # Clear the recorded revert target once we commit to reinstalling it, so we
    # don't keep offering to "revert" to the version we're now going back to.
    from settings_manager import get_settings
    _download_and_launch(
        parent, asset, release_url,
        on_before_launch=lambda: get_settings().set_previous_version(None),
    )


def show_up_to_date(parent, current_version):
    QMessageBox.information(parent, "Up to date",
                            f"You're running the latest version ({current_version}).")
