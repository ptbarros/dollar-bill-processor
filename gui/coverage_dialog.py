"""
Coverage Check (Core vs a reference set).

After processing a strap with a lean "Core" selection enabled, this re-classifies
the same serials against a REFERENCE set and shows what Core would miss — so you
can trust a lean set loses nothing that matters and see which patterns to add.

The reference defaults to the full installed library, but that includes very
loose patterns (e.g. any two-of-a-kind) that flag almost every bill, which
drowns the diff. Pick a saved preset (e.g. your "Originals" working set) as the
reference instead for a meaningful comparison.

No re-scanning: it reuses the already-processed results (serial + seal/gas
metadata) and just re-runs the pattern engine.
"""

from collections import Counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTableWidget,
    QTableWidgetItem, QGroupBox, QDialogButtonBox, QHeaderView,
    QAbstractItemView, QPushButton, QFileDialog, QApplication, QWidget,
)

FULL_LIBRARY = "Full installed library (all patterns)"


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


def compute_coverage(engine, results, reference_names=None):
    """Diff the current (Core) selection against a reference over the results.
    reference_names=None -> the whole installed library; otherwise that set.

    Classifies in two batch passes (Core, then reference) toggling the enabled
    set once per pass rather than per bill, so a 1000-bill run stays quick."""
    items = []
    for r in results:
        serial = (r.get("serial") or "").strip()
        if len("".join(c for c in serial if c.isdigit())) != 8:
            continue
        items.append((serial, _metadata(r, engine)))

    lp = engine.lua_patterns
    snapshot = {n: p.enabled for n, p in lp.items()}
    try:
        # Core pass uses the current (live) enabled selection.
        core_res = [set(engine.classify_simple(s, md)) for s, md in items]
        # Reference pass: whole library, or the given subset.
        for n, p in lp.items():
            p.enabled = True if reference_names is None else (n in reference_names)
        ref_res = [set(engine.classify_simple(s, md)) for s, md in items]
    finally:
        for n, p in lp.items():
            p.enabled = snapshot.get(n, p.enabled)

    core_fancy = ref_fancy = 0
    missed = []
    gap_patterns = Counter()
    for (serial, _), core, ref in zip(items, core_res, ref_res):
        if core:
            core_fancy += 1
        if ref:
            ref_fancy += 1
        only = ref - core
        if only:
            for p in only:
                gap_patterns[p] += 1
            if not core:
                missed.append((serial, sorted(only)))
    return {
        "total": len(items), "core_fancy": core_fancy, "ref_fancy": ref_fancy,
        "missed": missed, "gap_patterns": gap_patterns.most_common(),
    }


class CoverageDialog(QDialog):
    def __init__(self, engine, results, presets=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.results = results
        self.presets = presets or {}   # name -> set of enabled pattern names
        self.setWindowTitle("Coverage Check — Core vs Reference")
        self.resize(700, 640)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Compare against:"))
        self.ref_combo = QComboBox()
        self.ref_combo.addItem(FULL_LIBRARY)
        for name in sorted(self.presets):
            self.ref_combo.addItem(name)
        self.ref_combo.setToolTip(
            "The set Core is measured against. The full library flags almost every "
            "bill (loose patterns like any two-of-a-kind) — pick a saved preset "
            "such as your 'Originals' set for a meaningful comparison.")
        self.ref_combo.currentIndexChanged.connect(self._recompute)
        top.addWidget(self.ref_combo, 1)
        layout.addLayout(top)

        self.body = QWidget()
        layout.addWidget(self.body, 1)

        row = QHBoxLayout()
        save_btn = QPushButton("Save Report…")
        save_btn.clicked.connect(self._save)
        row.addWidget(save_btn)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)

        self._recompute()

    def _reference_names(self):
        name = self.ref_combo.currentText()
        if name == FULL_LIBRARY:
            return None
        return self.presets.get(name, set())

    def _recompute(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.rep = compute_coverage(self.engine, self.results, self._reference_names())
        finally:
            QApplication.restoreOverrideCursor()
        self._render()

    def _render(self):
        # rebuild body
        old = self.body.layout()
        if old is not None:
            QWidget().setLayout(old)  # detach the old layout
        lay = QVBoxLayout(self.body)
        rep = self.rep
        ref_label = self.ref_combo.currentText()
        missed_n = len(rep["missed"])
        pct = (rep["core_fancy"] / rep["ref_fancy"] * 100) if rep["ref_fancy"] else 100.0
        verdict = ("Core caught everything the reference would."
                   if missed_n == 0 else
                   f"Core would skip {missed_n} bill{'s' if missed_n != 1 else ''} "
                   f"the reference flags.")
        head = QLabel(
            f"<b>{verdict}</b><br>Checked <b>{rep['total']}</b> bills · "
            f"Core flagged <b>{rep['core_fancy']}</b>, reference "
            f"(<i>{ref_label}</i>) flagged <b>{rep['ref_fancy']}</b> "
            f"(Core covers {pct:.1f}% of the reference's hits).")
        head.setWordWrap(True); head.setTextFormat(Qt.RichText)
        lay.addWidget(head)

        miss_box = QGroupBox(f"Bills Core skipped ({missed_n}) — caught only by the reference")
        mb = QVBoxLayout(miss_box)
        t1 = self._table(["Serial", "Caught by (reference-only patterns)"])
        self._fill(t1, [(s, ", ".join(ps)) for s, ps in rep["missed"]])
        mb.addWidget(t1); lay.addWidget(miss_box, 2)

        gap_box = QGroupBox("Patterns Core is missing vs the reference (by bills they'd add)")
        gb = QVBoxLayout(gap_box)
        t2 = self._table(["Pattern", "Bills it would add"])
        self._fill(t2, [(p, str(n)) for p, n in rep["gap_patterns"]], numeric_col=1)
        gb.addWidget(t2); lay.addWidget(gap_box, 1)

    def _table(self, headers):
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    def _fill(self, table, rows, numeric_col=None):
        table.setRowCount(len(rows) or 1)
        if not rows:
            cell = QTableWidgetItem("— none —"); cell.setForeground(Qt.gray)
            table.setItem(0, 0, cell); return
        for i, cells in enumerate(rows):
            for j, val in enumerate(cells):
                item = QTableWidgetItem(str(val))
                if numeric_col is not None and j == numeric_col:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, j, item)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Coverage Report", "coverage_report.txt", "Text (*.txt)")
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"
        rep = self.rep
        lines = ["Dollar Detective — Core vs Reference coverage check",
                 f"Reference: {self.ref_combo.currentText()}", "",
                 f"Bills checked: {rep['total']}",
                 f"Core flagged: {rep['core_fancy']}   Reference: {rep['ref_fancy']}", "",
                 f"Bills Core skipped ({len(rep['missed'])}):"]
        lines += [f"  {s}\t{', '.join(ps)}" for s, ps in rep["missed"]]
        lines += ["", "Patterns Core is missing vs the reference (bills they'd add):"]
        lines += [f"  {n}\t{p}" for p, n in rep["gap_patterns"]]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass
