# Refactor plan: `dollar_detective/` package

Goal: collapse the ~27 loose `.py` modules in the repo root into a single
`dollar_detective/` package, so the GitHub repo page shows ~a dozen root entries
instead of ~57 and the README lands near the top.

**Golden rule:** do it all on a branch, never touch `main`, and treat it as
unfinished until *both* a CI build and the web sync are re-verified. Phases 1–6
are an afternoon; Phase 7 (verify) is the real work, because the two highest-risk
changes (Phase 4 and Phase 6) only fail at runtime/build, not at import.

---

## Phase 0 — Prep

1. **Clean the working tree first.** Root has ~25 untracked scratch files
   (`chatgpt.txt`, `error.txt`, `*.zip`, …). Commit, delete, or gitignore them so
   the `git mv` diff is reviewable. A `git mv`-heavy refactor on a messy tree is
   unreadable in review.
2. `git switch -c refactor/package-layout`

## Phase 1 — Decide the partition

**Stay at root** (entry scripts + spec hooks — moving them would change `.spec`
entry/hook paths):
- `run_gui.py`, `crop_tool.py` (PyInstaller entries)
- `pyi_rth_cv2.py`, `rthook_force_cuda.py` (spec `runtime_hooks`)
- `*.spec`, `requirements*.txt`, `config.yaml`, `best.onnx`, `best.pt`,
  `README.md`, `LICENSE`, `CLAUDE.md`, `.gitignore`

**Move into `dollar_detective/`** (the ~27 library modules):
`ai_pattern_chat, backup_manager, batch_naming, correction_manager, crop_batch,
crop_geometry, crop_preview, crop_samples, debug_logger, example_corpus, ledger,
ocr_backend, pattern_bundle, pattern_corpus, pattern_engine_v3, pattern_sandbox,
process_production, resource_path, review_bundle, serial_overlay,
session_recovery, settings_manager, system_info, updater, verify_imports,
version, yolo_backend`

**Relocate out of root** (not into the package): `test_pattern_chat.py` →
`tests/`; `train_yolo.py` → `tools/` (optional dev script).

**Leave as-is:** `gui/` stays its own top-level package (first pass). Moving it in
is strictly cleaner but adds the most import churn and a `gui.updater_ui`
hiddenimport change — defer it.

## Phase 2 — Move the files (history-preserving)

```bash
mkdir dollar_detective
touch dollar_detective/__init__.py        # optionally: __version__ re-export
git mv pattern_engine_v3.py pattern_sandbox.py process_production.py \
       resource_path.py settings_manager.py version.py ocr_backend.py \
       yolo_backend.py serial_overlay.py updater.py verify_imports.py \
       ai_pattern_chat.py backup_manager.py batch_naming.py correction_manager.py \
       crop_batch.py crop_geometry.py crop_preview.py crop_samples.py \
       debug_logger.py example_corpus.py ledger.py pattern_bundle.py \
       pattern_corpus.py review_bundle.py session_recovery.py system_info.py \
       dollar_detective/
git mv test_pattern_chat.py tests/
git mv train_yolo.py tools/     # optional
```

## Phase 3 — Rewrite imports (codemod, ~50 lines across the repo)

Use absolute `dollar_detective.`-qualified imports everywhere (works from the root
entry scripts, `gui/`, `tools/`, `tests/`, and inside the package, as long as the
repo root is on `sys.path` — which `run_gui.py` already ensures and pytest/CI get
for free). A script, not sed, so word boundaries are safe:

```python
# tools/_codemod_pkg.py  — run once from repo root, then delete
import re, pathlib
PKG = "dollar_detective"
MOVED = {"ai_pattern_chat","backup_manager","batch_naming","correction_manager",
  "crop_batch","crop_geometry","crop_preview","crop_samples","debug_logger",
  "example_corpus","ledger","ocr_backend","pattern_bundle","pattern_corpus",
  "pattern_engine_v3","pattern_sandbox","process_production","resource_path",
  "review_bundle","serial_overlay","session_recovery","settings_manager",
  "system_info","updater","verify_imports","version","yolo_backend"}
SKIP = {"venv",".venv","build","dist","__pycache__",".git"}
for py in pathlib.Path(".").rglob("*.py"):
    if set(py.parts) & SKIP: continue
    t = t0 = py.read_text(encoding="utf-8")
    for m in MOVED:
        t = re.sub(rf'^(\s*)from {m} import ', rf'\1from {PKG}.{m} import ', t, flags=re.M)
        t = re.sub(rf'^(\s*)import {m} as ',   rf'\1from {PKG} import {m} as ', t, flags=re.M)
        t = re.sub(rf'^(\s*)import {m}\s*$',   rf'\1from {PKG} import {m}', t, flags=re.M)
    if t != t0: py.write_text(t, encoding="utf-8"); print("rewrote", py)
```

Then **eyeball by hand** for forms the regex won't catch: `import a, b` on one
line, and any *string* module names
(`grep -rn "pattern_engine_v3\|process_production" --include=*.py | grep -v import`
— `verify_imports.py` may list module names as strings).

## Phase 4 — Fix `resource_path` (the silent-breakage one)

`app_base()` is used by both desktop and the flat web vendor, with `patterns/` at
a *different* depth in each. So the fix is **not** `.parent.parent` (that would
break the vendor) — make the source-mode base auto-detect the dir that contains
`patterns/`. Apply to both `app_base()` and the source branch of `user_data_dir()`:

```python
def app_base() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent):          # package dir, then repo root
        if (cand / "patterns").is_dir():
            return cand
    return here.parent                         # fall back to repo root
```

## Phase 5 — PyInstaller specs

- `DollarDetective.spec`: `hiddenimports` — change `'updater'` →
  `'dollar_detective.updater'` (keep `'gui.updater_ui'`). Entry `['run_gui.py']`,
  `datas`, and `runtime_hooks` are **unchanged** (entries/hooks/data all still at
  root). Optionally add `pathex=['.']` defensively.
- `crop_tool.spec`: entry `['crop_tool.py']` unchanged. **Check its
  `hiddenimports` block** (lines ~53–63) and qualify any moved module names found
  there.

## Phase 6 — dd-web sync (`../dollardetective-web/scripts/sync-engine.sh`)

`sync-engine.sh` copies `pattern_engine_v3.py pattern_sandbox.py resource_path.py`
**flat** into `api/vendor/`, and `pattern_engine_v3.py` does
`from pattern_sandbox import …`. Two edits, both localized to the sync script —
the web app's own imports stay flat and unchanged:

```bash
# existence check:
[ -f "$DESKTOP/dollar_detective/pattern_engine_v3.py" ] || { echo "…"; exit 1; }

# the copy loop — strip the package prefix so api/vendor/ stays flat:
for f in "${MODULES[@]}"; do
  sed -E 's/\bfrom dollar_detective\.([A-Za-z0-9_]+) import /from \1 import /;
          s/\bfrom dollar_detective import /import /;
          s/\bimport dollar_detective\.([A-Za-z0-9_]+)\b/import \1/' \
    "$DESKTOP/dollar_detective/$f" > "$DEST/$f"
done
```

`patterns/` and `shipped_enabled.json` paths are unchanged (they stay at desktop
root). **Flag this change to the web side before landing** — standing rule for
anything touching the vendored set.

## Phase 7 — Verify (the part that actually matters)

```bash
# a. imports resolve (loud, instant)
venv/bin/python -c "import dollar_detective.pattern_engine_v3, dollar_detective.process_production, dollar_detective.gui"
PYTHONPATH=. venv/bin/python dollar_detective/pattern_engine_v3.py        # engine self-test

# b. the bundle's own import check + a real-inference smoke test
DBP_VERIFY_IMPORTS=1 venv/bin/python run_gui.py
DBP_SELFTEST=test_bills/<some>.jpg venv/bin/python run_gui.py            # PASS line

# c. test suite + GUI launch
venv/bin/python -m pytest -q
venv/bin/python run_gui.py
```

Then the builds — **in CI, not locally**:
```bash
git push -u origin refactor/package-layout
gh workflow run build.yml --ref refactor/package-layout -f editions=windows-directml   # ~6–7 min
gh workflow run crop-tool.yml --ref refactor/package-layout
```
Install the artifact, confirm it finds `best.pt`/`patterns/`/`config.yaml` (that's
the `resource_path` fix proving out) and reads a serial.

Finally the web:
```bash
cd ../dollardetective-web && DESKTOP_REPO=~/projects/dollar-bill-processor scripts/sync-engine.sh
grep -rn "dollar_detective" api/vendor/ && echo "LEAK — fix sed" || echo "clean"
# run its tests / local server
```

## Must-not-skip summary

- **Phase 4** — silent wrong data paths if missed.
- **Phase 6** — silent web breakage on next rebuild if missed.
Both only surface at runtime/build, which is why **Phase 7 is non-negotiable**.
