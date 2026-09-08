"""
Coverage Check (Core vs Full library).

After processing a strap with a lean "Core" pattern selection enabled, this runs
the SAME serials against the entire installed library and shows what Core would
have missed — so you can trust a lean set loses nothing that matters, and see
exactly which patterns to add to Core if it does.

No re-scanning: it re-classifies the already-processed results (which carry the
serial + seal/gas metadata) with the current selection vs. every pattern enabled.
"""

from collections import Counter

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QGroupBox, QDialogButtonBox, QHeaderView, QAbstractItemView, QPushButton,
    QFileDialog, QApplication,
)


def _metadata(result, engine):
    def f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    cont = f(result.get("seal_containment"), 100.0)
    return {
        "baseline_variance": f(result.get("baseline_variance")),
        "gas_pump_threshold": engine.get_gas_pump_threshold(),
        "seal_x": f(result.get("seal_x")),
        "seal_y": f(result.get("seal_y")),
        "seal_containment": cont if cont else 100.0,
        "series_year": result.get("series_year", "") or "",
        "front_plate": result.get("front_plate", "") or "",
        "back_plate": result.get("back_plate", "") or "",
    }


def compute_coverage(engine, results):
    """Return a coverage report dict comparing the current selection (Core) with
    the full library over the given result rows."""
    total = 0
    core_fancy = 0
    full_fancy = 0
    missed = []          # (serial, [patterns full caught]) — Core said NOT fancy
    extra = []           # (serial, [extra patterns]) — Core fancy but full caught more
    gap_patterns = Counter()   # full-only pattern -> # bills it added
    for r in results:
        serial = (r.get("serial") or "").strip()
        digits = "".join(c for c in serial if c.isdigit())
        if len(digits) != 8:
            continue
        md = _metadata(r, engine)
        try:
            core = set(engine.classify_simple(serial, md))
            full = set(engine.classify_full(serial, md))
        except Exception:
            continue
        total += 1
        if core:
            core_fancy += 1
        if full:
            full_fancy += 1
        only = full - core
        if only:
            for p in only:
                gap_patterns[p] += 1
            if not core:
                missed.append((serial, sorted(only)))
            else:
                extra.append((serial, sorted(only)))
    return {
        "total": total,
        "core_fancy": core_fancy,
        "full_fancy": full_fancy,
        "missed": missed,
        "extra": extra,
        "gap_patterns": gap_patterns.most_common(),
    }


class CoverageDialog(QDialog):
    def __init__(self, engine, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coverage Check — Core vs Full Library")
        self.resize(680, 620)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.rep = compute_coverage(engine, results)
        finally:
            QApplication.restoreOverrideCursor()
        self._build()

    def _build(self):
        rep = self.rep
        layout = QVBoxLayout(self)

        total = rep["total"]
        missed_n = len(rep["missed"])
        extra_n = len(rep["extra"])
        pct = (rep["core_fancy"] / rep["full_fancy"] * 100) if rep["full_fancy"] else 100.0
        verdict = ("Core caught everything the full library would." if missed_n == 0
                   else f"Core would have skipped {missed_n} bill"
                        f"{'s' if missed_n != 1 else ''} the full library flags.")
        head = QLabel(
            f"<b>{verdict}</b><br>"
            f"Checked <b>{total}</b> bills · Core flagged <b>{rep['core_fancy']}</b>, "
            f"full library would flag <b>{rep['full_fancy']}</b> "
            f"(Core covers {pct:.1f}% of the full library's hits).<br>"
            f"<span style='color:gray'>{extra_n} more bills are flagged by both, but the "
            f"full library adds extra pattern labels to them.</span>"
        )
        head.setWordWrap(True)
        head.setTextFormat(Qt.RichText)
        layout.addWidget(head)

        # Bills Core skipped entirely — the "am I missing keepers?" answer
        miss_box = QGroupBox(f"Bills Core skipped ({missed_n}) — caught only by the full library")
        mb = QVBoxLayout(miss_box)
        self.miss_table = self._make_table(["Serial", "Caught by (full-only patterns)"])
        self._fill(self.miss_table, [(s, ", ".join(ps)) for s, ps in rep["missed"]])
        mb.addWidget(self.miss_table)
        layout.addWidget(miss_box, 2)

        # Which patterns account for the gap → candidates to add to Core
        gap_box = QGroupBox("Patterns to consider adding to Core (by bills they'd add)")
        gb = QVBoxLayout(gap_box)
        self.gap_table = self._make_table(["Pattern", "Bills it would add"])
        self._fill(self.gap_table, [(p, str(n)) for p, n in rep["gap_patterns"]],
                   numeric_col=1)
        gb.addWidget(self.gap_table)
        layout.addWidget(gap_box, 1)

        row = QHBoxLayout()
        save_btn = QPushButton("Save Report…")
        save_btn.clicked.connect(self._save)
        row.addWidget(save_btn)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        row.addWidget(buttons)
        layout.addLayout(row)

    def _make_table(self, headers):
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    def _fill(self, table, rows, numeric_col=None):
        table.setRowCount(len(rows))
        for i, cells in enumerate(rows):
            for j, val in enumerate(cells):
                item = QTableWidgetItem(str(val))
                if numeric_col is not None and j == numeric_col:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, j, item)
        if not rows:
            table.setRowCount(1)
            cell = QTableWidgetItem("— none —")
            cell.setForeground(Qt.gray)
            table.setItem(0, 0, cell)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Coverage Report", "coverage_report.txt", "Text (*.txt)")
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"
        rep = self.rep
        lines = ["Dollar Detective — Core vs Full coverage check", ""]
        lines.append(f"Bills checked: {rep['total']}")
        lines.append(f"Core flagged: {rep['core_fancy']}   Full library: {rep['full_fancy']}")
        lines.append("")
        lines.append(f"Bills Core skipped ({len(rep['missed'])}):")
        for s, ps in rep["missed"]:
            lines.append(f"  {s}\t{', '.join(ps)}")
        lines.append("")
        lines.append("Patterns to consider adding to Core (bills they'd add):")
        for p, n in rep["gap_patterns"]:
            lines.append(f"  {n}\t{p}")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass
