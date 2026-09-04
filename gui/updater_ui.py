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

    box = QMessageBox(parent)
    box.setWindowTitle("Update available")
    box.setIcon(QMessageBox.Information)
    box.setText(f"Dollar Detective {info.latest_version} is available.\n"
                f"You have {info.current_version}.")
    if can_autoinstall:
        box.setInformativeText(
            "Download and install it now? The app will close to finish installing.")
        accept = box.addButton("Download && Install", QMessageBox.AcceptRole)
    else:
        box.setInformativeText("Open the download page in your browser?")
        accept = box.addButton("Open Download Page", QMessageBox.AcceptRole)
    box.addButton("Later", QMessageBox.RejectRole)
    box.exec()
    if box.clickedButton() is not accept:
        return

    if not can_autoinstall:
        webbrowser.open(info.release_url)
        return

    # Windows: download the matching installer with a progress dialog.
    prog = QProgressDialog("Downloading update…", "Cancel", 0, 100, parent)
    prog.setWindowTitle("Downloading update")
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
    prog.reset()
    if state["cancelled"]:
        return
    t.wait()

    path = state["path"]
    if not path:
        QMessageBox.warning(parent, "Download failed",
                            "Couldn't download the update. Opening the download page instead.")
        webbrowser.open(info.release_url)
        return

    if updater.launch_installer_and_exit(path):
        QApplication.quit()
    else:
        webbrowser.open(info.release_url)


def show_up_to_date(parent, current_version):
    QMessageBox.information(parent, "Up to date",
                            f"You're running the latest version ({current_version}).")
