"""
Ask Claude about this pattern — a multi-turn chat window (Pattern Manager).

One unified chat: it answers questions / compares patterns AND writes or fixes
pattern scripts (the merged prompt lives in ai_pattern_chat.py). The selected
pattern is the context by default; "Compare to…" pulls in another pattern so the
model has its Lua too. The API call runs on a worker thread so the UI never
freezes. Phase 1 is text only — live overlay rendering of suggested styles is
a later phase.
"""
import html
import re
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTextBrowser, QTextEdit, QMessageBox,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import ai_pattern_chat


class _ChatWorker(QThread):
    """Runs one session.ask() off the UI thread."""
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, session, user_text, context):
        super().__init__()
        self._session = session
        self._user_text = user_text
        self._context = context

    def run(self):
        result = self._session.ask(self._user_text, context=self._context)
        if result.success:
            self.finished_ok.emit(result.text)
        else:
            self.failed.emit(result.error)


class PatternChatDialog(QDialog):
    """Chat about the selected pattern (optionally comparing to others)."""

    def __init__(self, engine, settings, pattern_name, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.settings = settings
        self.base_pattern = pattern_name
        # Patterns currently in context (base first, then any compare-to picks).
        self.context_names = [pattern_name]
        # Whether the pattern context needs (re)sending on the next turn.
        self._context_dirty = True
        self._worker = None

        provider, api_key, model = ai_pattern_chat.resolve_provider(settings)
        self.session = ai_pattern_chat.PatternChatSession(
            provider=provider, api_key=api_key, model=model)

        display = self._display(pattern_name)
        self.setWindowTitle(f"Ask Claude — {display}")
        self.resize(680, 620)
        self._build_ui(display, provider, model)

    # --- UI ---------------------------------------------------------------
    def _build_ui(self, display, provider, model):
        layout = QVBoxLayout(self)

        header = QLabel(
            f"Chatting about <b>{html.escape(display)}</b>. Ask what it does, compare it "
            f"to another pattern, or ask for a change to the script.")
        header.setWordWrap(True)
        layout.addWidget(header)

        # Compare-to row
        cmp_row = QHBoxLayout()
        cmp_row.addWidget(QLabel("Compare to:"))
        self.compare_combo = QComboBox()
        self.compare_combo.setMinimumWidth(280)
        self._populate_compare_combo()
        cmp_row.addWidget(self.compare_combo, 1)
        self.add_compare_btn = QPushButton("Add")
        self.add_compare_btn.setToolTip("Add this pattern to the conversation so Claude can see its script too")
        self.add_compare_btn.clicked.connect(self._add_compare)
        cmp_row.addWidget(self.add_compare_btn)
        layout.addLayout(cmp_row)

        self.context_label = QLabel()
        self.context_label.setWordWrap(True)
        self.context_label.setStyleSheet("color: #555;")
        self._refresh_context_label()
        layout.addWidget(self.context_label)

        # Transcript
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        layout.addWidget(self.transcript, 1)

        # Status line
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #1565C0;")
        layout.addWidget(self.status_label)

        # Input row
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText(
            "Ask a question…  (Ctrl+Enter to send)")
        self.input_edit.setMaximumHeight(90)
        self.input_edit.installEventFilter(self)
        layout.addWidget(self.input_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.send_btn = QPushButton("Send")
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self._send)
        btn_row.addWidget(self.send_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        if not self.session.api_key:
            self._append_system(
                "No AI provider is configured. Add an API key in Settings → AI.")
            self.send_btn.setEnabled(False)
            self.input_edit.setEnabled(False)
        else:
            self._append_system(
                f"Ready ({provider}, {model}). Ask away — the first message includes "
                f"this pattern's script automatically.")

    def _populate_compare_combo(self):
        """All other loaded patterns, by display name, for the compare-to picker."""
        self.compare_combo.clear()
        items = []
        for name in self.engine.lua_patterns:
            if name == self.base_pattern:
                continue
            items.append((self._display(name), name))
        items.sort(key=lambda t: t[0].lower())
        for disp, name in items:
            self.compare_combo.addItem(disp, name)

    # --- helpers ----------------------------------------------------------
    def _display(self, name):
        info = self.engine.get_pattern_info(name)
        if info and info.get("display_name"):
            return info["display_name"]
        return name

    def _refresh_context_label(self):
        names = ", ".join(self._display(n) for n in self.context_names)
        self.context_label.setText(f"In this conversation: {names}")

    def eventFilter(self, obj, event):
        # Ctrl+Enter (or Cmd+Enter) sends.
        if obj is self.input_edit and event.type() == event.Type.KeyPress:
            if (event.key() in (Qt.Key_Return, Qt.Key_Enter)
                    and event.modifiers() & Qt.ControlModifier):
                self._send()
                return True
        return super().eventFilter(obj, event)

    def _add_compare(self):
        name = self.compare_combo.currentData()
        if not name or name in self.context_names:
            return
        self.context_names.append(name)
        self._context_dirty = True  # send the updated context next turn
        self._refresh_context_label()
        self._append_system(f"Added {self._display(name)} to the conversation.")

    # --- sending ----------------------------------------------------------
    def _send(self):
        if self._worker is not None:
            return  # a request is already in flight
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        if not self.session.api_key:
            QMessageBox.warning(self, "Not configured",
                                "Add an API key in Settings → AI first.")
            return

        context = ""
        if self._context_dirty:
            context = ai_pattern_chat.build_pattern_context(self.engine, self.context_names)
            self._context_dirty = False

        self._append_message("You", text)
        self.input_edit.clear()
        self._set_busy(True)

        self._worker = _ChatWorker(self.session, text, context)
        self._worker.finished_ok.connect(self._on_answer)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._worker_done)
        self._worker.start()

    def _on_answer(self, text):
        self._append_message("Claude", text)

    def _on_error(self, error):
        self._append_system(f"⚠ {error}")

    def _worker_done(self):
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy):
        self.send_btn.setEnabled(not busy)
        self.input_edit.setReadOnly(busy)
        self.add_compare_btn.setEnabled(not busy)
        self.status_label.setText("Claude is thinking…" if busy else "")

    # --- transcript rendering --------------------------------------------
    def _append_system(self, text):
        self.transcript.append(
            f'<p style="color:#777;font-style:italic;margin:6px 0;">{html.escape(text)}</p>')
        self._scroll_bottom()

    def _append_message(self, who, text):
        color = "#1565C0" if who == "You" else "#2E7D32"
        self.transcript.append(
            f'<p style="margin:8px 0 2px 0;"><b style="color:{color};">{who}</b></p>'
            f'{self._format_body(text)}')
        self._scroll_bottom()

    def _format_body(self, text):
        """Escape HTML, render ```code``` fences as <pre>, keep line breaks."""
        out = []
        for i, chunk in enumerate(re.split(r"```(?:lua)?\n?", text)):
            if i % 2 == 1:  # inside a fence
                out.append(
                    '<pre style="background:#f4f4f4;border:1px solid #ddd;padding:6px;'
                    'white-space:pre-wrap;font-family:monospace;">'
                    f'{html.escape(chunk.rstrip())}</pre>')
            elif chunk.strip():
                body = html.escape(chunk.strip()).replace("\n", "<br>")
                out.append(f'<div style="margin:0 0 6px 0;">{body}</div>')
        return "".join(out)

    def _scroll_bottom(self):
        sb = self.transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        # Let an in-flight request finish quietly rather than crash on teardown.
        if self._worker is not None:
            self._worker.wait(3000)
        super().closeEvent(event)
