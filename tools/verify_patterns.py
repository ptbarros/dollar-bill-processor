#!/usr/bin/env python3
"""
verify_patterns.py - Cross-check the 104 implemented Green Guide Lua patterns
against tggfsn.ods.

Checks per pattern:
  1. BookRef field in Lua matches ODS CS# for that pattern name
  2. Example serials in the Lua header all match (matched=True) when run through engine
  3. ODS bare-8-digit example serials (col 7) all match when run through engine
  4. Side-by-side Lua Description vs ODS Description (human review, no pass/fail)

Usage:
    python3 tools/verify_patterns.py
    python3 tools/verify_patterns.py --examples-only
    python3 tools/verify_patterns.py --failures-only
"""

import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or tools/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from pattern_engine_v3 import PatternEngineV3

# ---------------------------------------------------------------------------
# ODS constants (0-based column indices, verified 2026-02-23)
# ---------------------------------------------------------------------------
ODS_PATH = Path.home() / 'projects' / 'tggfsn.ods'

COL_SKIP      = 3   # 'x' = do not implement
COL_CREATED   = 4   # 'x' = implemented as Lua
COL_CS        = 5   # CS number (bare integer, e.g. "100")
COL_NAME      = 6   # Display name, e.g. "CS-Triple"
COL_EXAMPLES  = 7   # Positional examples in M xx M format
COL_DESC      = 8   # Book prose description

NS_TABLE = 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'
NS_TEXT  = 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'

# ---------------------------------------------------------------------------
# ODS helpers
# ---------------------------------------------------------------------------

def _get_cell_text(cell: ET.Element) -> str:
    parts = [p.text for p in cell.findall(f'.//{{{NS_TEXT}}}p') if p.text]
    return ' | '.join(parts) if parts else ''


def _get_row_cells(row: ET.Element, n_cols: int) -> list[str]:
    cols: list[str] = []
    for cell in row.findall(f'{{{NS_TABLE}}}table-cell'):
        rep = cell.get(f'{{{NS_TABLE}}}number-columns-repeated')
        count = int(rep) if rep else 1
        text = _get_cell_text(cell)
        cols.extend([text] * count)
        if len(cols) >= n_cols:
            break
    padding = max(0, n_cols - len(cols))
    return cols[:n_cols] + [''] * padding


def _extract_m_serials(text: str) -> list[str]:
    """
    Extract testable 8-digit serials from text containing M-marker examples.

    Handles:
    - Standard:      M 12345678 M
    - Spaced digits: M 1 2 3 4 5 6 7 8 M  → 12345678
    - Skips any example that contains 'x' (positional format, not testable)
    """
    results: list[str] = []
    # Match anything between M ... M (non-greedy)
    for raw in re.findall(r'M\s+([\dx\s]+?)\s*M', text, re.IGNORECASE):
        # Collapse internal spaces to get compact form
        compact = re.sub(r'\s+', '', raw)
        # Skip if it still has x wildcards
        if re.search(r'[xX]', compact):
            continue
        # Must be exactly 8 digits
        if re.fullmatch(r'\d{8}', compact):
            results.append(compact)
    return results


def load_ods_implemented(ods_path: Path) -> dict[str, dict]:
    """
    Return dict keyed by "CS-NNN" for every row where Created='x' and Skip=''.

    Each value is:
        {
            'cs_key':      'CS-100',
            'name':        'CS-Triple',
            'examples':    ['11100000', ...],   # testable 8-digit serials; may be empty
            'desc':        'Book prose...',
            'ex_source':   'examples' | 'description' | 'none',
        }

    When the Examples column (col 7) yields no testable serials, the Description
    column (col 8) is also scanned for M-marker examples.
    """
    with zipfile.ZipFile(ods_path) as z:
        content = z.read('content.xml').decode('utf-8')

    root = ET.fromstring(content)
    ns = {'table': NS_TABLE}
    sheet = root.findall('.//table:table', ns)[0]
    rows = sheet.findall('table:table-row', ns)[1:]   # skip header

    result: dict[str, dict] = {}
    for row in rows:
        cells = _get_row_cells(row, 10)
        if cells[COL_CREATED] != 'x' or cells[COL_SKIP] == 'x':
            continue
        cs_num = cells[COL_CS].strip()
        if not cs_num:
            continue
        cs_key = f'CS-{cs_num}'
        name   = cells[COL_NAME].strip()
        desc   = cells[COL_DESC].strip()

        # Try examples column first
        bare_serials = _extract_m_serials(cells[COL_EXAMPLES])
        ex_source = 'examples'

        # Fall back to description column when examples column is empty/positional-only
        if not bare_serials and desc:
            from_desc = _extract_m_serials(desc)
            if from_desc:
                bare_serials = from_desc
                ex_source = 'description'

        if not bare_serials:
            ex_source = 'none'

        result[cs_key] = {
            'cs_key':    cs_key,
            'name':      name,
            'examples':  bare_serials,
            'desc':      desc,
            'ex_source': ex_source,
        }
    return result


# ---------------------------------------------------------------------------
# Lua header parser
# ---------------------------------------------------------------------------

def parse_lua_header(script: str) -> dict[str, str | list[str]]:
    """
    Return dict with lowercase keys parsed from the --[[ ... --]] header.
    'examples' is a list[str]; all other values are str.
    """
    meta: dict[str, str | list[str]] = {}
    m = re.search(r'--\[\[(.*?)--\]\]', script, re.DOTALL)
    if not m:
        return meta
    for line in m.group(1).split('\n'):
        km = re.match(r'\s*(\w+):\s*(.+)', line)
        if not km:
            continue
        key   = km.group(1).lower()
        value = km.group(2).strip()
        if key == 'examples':
            try:
                import json
                meta[key] = json.loads(value)
            except Exception:
                meta[key] = [value]
        else:
            meta[key] = value
    return meta


# ---------------------------------------------------------------------------
# Scan Green Guide Lua files
# ---------------------------------------------------------------------------

def load_lua_patterns(project_dir: Path) -> list[dict]:
    """
    Scan patterns/The Green Guide/*.lua and return list of dicts:
        {
            'file':       Path,
            'pattern':    'CS_TRIPLE',      # Pattern: header field
            'displayname':'CS-Triple',
            'description':'...',
            'bookref':    'CS-100',
            'examples':   ['00033300', ...],
        }
    """
    gg_dir = project_dir / 'patterns' / 'The Green Guide'
    results = []
    for lua_file in sorted(gg_dir.glob('*.lua')):
        script = lua_file.read_text(encoding='utf-8')
        meta = parse_lua_header(script)
        if not meta:
            continue
        results.append({
            'file':        lua_file,
            'pattern':     meta.get('pattern', lua_file.stem.upper()),
            'displayname': meta.get('displayname', ''),
            'description': meta.get('description', ''),
            'bookref':     meta.get('bookref', ''),
            'examples':    meta.get('examples', []),
        })
    return results


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ExResult:
    def __init__(self, serial: str, matched: bool, error: str = ''):
        self.serial  = serial
        self.matched = matched
        self.error   = error

    def __repr__(self):
        status = 'OK' if self.matched else ('ERR' if self.error else 'FAIL')
        return f'{status}:{self.serial}'


class PatternReport:
    PASS = 'PASS'
    WARN = 'WARN'
    FAIL = 'FAIL'

    def __init__(self, lua: dict, ods: dict | None):
        self.lua          = lua
        self.ods          = ods
        self.bookref_ok   = True
        self.lua_results  : list[ExResult] = []
        self.ods_results  : list[ExResult] = []
        self.status       = self.WARN

    def compute_status(self):
        all_results = self.lua_results + self.ods_results
        if not self.bookref_ok:
            self.status = self.FAIL
            return
        any_fail = any(not r.matched for r in all_results)
        if any_fail:
            self.status = self.FAIL
        elif all_results:
            self.status = self.PASS
        else:
            self.status = self.WARN


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------

def run_verification(engine: PatternEngineV3,
                     lua_patterns: list[dict],
                     ods_data: dict[str, dict]) -> list[PatternReport]:
    reports = []
    for lua in lua_patterns:
        bookref = lua['bookref']
        ods = ods_data.get(bookref)

        report = PatternReport(lua=lua, ods=ods)

        # 1. BookRef check
        if not bookref:
            report.bookref_ok = False
        elif ods is None:
            report.bookref_ok = False   # no ODS entry for this BookRef

        # 2. Run Lua header examples
        pattern_name = str(lua['pattern'])
        for serial in lua['examples']:
            if not isinstance(serial, str) or len(serial) != 8:
                continue
            try:
                res = engine.execute_pattern(pattern_name, serial)
                report.lua_results.append(ExResult(serial, res.matched))
            except Exception as e:
                report.lua_results.append(ExResult(serial, False, str(e)))

        # 3. Run ODS bare-serial examples
        if ods:
            for serial in ods['examples']:
                try:
                    res = engine.execute_pattern(pattern_name, serial)
                    report.ods_results.append(ExResult(serial, res.matched))
                except Exception as e:
                    report.ods_results.append(ExResult(serial, False, str(e)))

        report.compute_status()
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

TERM_RESET  = '\033[0m'
TERM_RED    = '\033[31m'
TERM_GREEN  = '\033[32m'
TERM_YELLOW = '\033[33m'
TERM_BOLD   = '\033[1m'
TERM_DIM    = '\033[2m'

def _colored(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f'{color}{text}{TERM_RESET}'
    return text


def _status_tag(status: str) -> str:
    if status == PatternReport.PASS:
        return _colored('[PASS]', TERM_GREEN)
    if status == PatternReport.FAIL:
        return _colored('[FAIL]', TERM_RED)
    return _colored('[WARN]', TERM_YELLOW)


def _result_list(results: list[ExResult], label: str) -> list[str]:
    if not results:
        return []
    lines = []
    all_ok = all(r.matched for r in results)
    summary = _colored('all ok', TERM_GREEN) if all_ok else _colored('FAILURES', TERM_RED)
    lines.append(f'  {label} ({len(results)} serials — {summary}):')
    for r in results:
        if r.matched:
            tag = _colored('ok  ', TERM_GREEN)
        elif r.error:
            tag = _colored('ERR ', TERM_RED)
        else:
            tag = _colored('FAIL', TERM_RED)
        lines.append(f'    {tag}  {r.serial}' + (f'  [{r.error}]' if r.error else ''))
    return lines


def _wrap(text: str, width: int = 70, indent: str = '    ') -> str:
    if not text:
        return f'{indent}(none)'
    words = text.split()
    lines: list[str] = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width and current != indent:
            lines.append(current.rstrip())
            current = indent + word + ' '
        else:
            current += word + ' '
    if current.strip():
        lines.append(current.rstrip())
    return '\n'.join(lines)


def print_report(reports: list[PatternReport],
                 examples_only: bool = False,
                 failures_only: bool = False) -> None:
    total  = len(reports)
    passed = sum(1 for r in reports if r.status == PatternReport.PASS)
    warned = sum(1 for r in reports if r.status == PatternReport.WARN)
    failed = sum(1 for r in reports if r.status == PatternReport.FAIL)
    no_ods = sum(1 for r in reports if r.ods is None)

    for report in reports:
        if failures_only and report.status != PatternReport.FAIL:
            continue

        lua  = report.lua
        ods  = report.ods
        name = lua['displayname'] or str(lua['pattern'])

        header = f"{_status_tag(report.status)} {_colored(name, TERM_BOLD)}"
        bookref_info = lua['bookref'] or _colored('(missing)', TERM_RED)
        header += f'  [{bookref_info}]'
        if ods is None:
            header += _colored('  — no ODS match', TERM_YELLOW)
        print(header)

        # BookRef mismatch detail
        if not report.bookref_ok and lua['bookref'] and ods is None:
            print(f'  {_colored("BookRef not found in ODS:", TERM_RED)} {lua["bookref"]}')

        # Example results
        ods_label = 'ODS examples'
        if ods and ods.get('ex_source') == 'description':
            ods_label = 'ODS examples (from description)'
        lines = _result_list(report.lua_results, 'Lua examples')
        lines += _result_list(report.ods_results, ods_label)
        if not lines and not report.lua_results and not report.ods_results:
            print(f'  {_colored("(no examples to test)", TERM_DIM)}')
        for l in lines:
            print(l)

        # Descriptions (skip in --examples-only mode)
        if not examples_only:
            lua_desc = str(lua.get('description', ''))
            ods_desc = ods['desc'] if ods else ''
            print(f'  {_colored("Lua desc:", TERM_DIM)}')
            print(_wrap(lua_desc))
            print(f'  {_colored("ODS desc:", TERM_DIM)}')
            print(_wrap(ods_desc))

        print()

    # Summary
    print(_colored('─' * 60, TERM_DIM))
    print(f'Patterns checked : {total}')
    print(f'  {_colored("PASS", TERM_GREEN)}  : {passed}')
    print(f'  {_colored("WARN", TERM_YELLOW)}  : {warned}  (no examples to test)')
    print(f'  {_colored("FAIL", TERM_RED)}  : {failed}')
    print(f'No ODS match     : {no_ods}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    examples_only = '--examples-only' in args
    failures_only = '--failures-only' in args

    print('Loading ODS ...', flush=True)
    ods_data = load_ods_implemented(ODS_PATH)
    print(f'  {len(ods_data)} implemented patterns in ODS')

    print('Loading Lua patterns ...', flush=True)
    lua_patterns = load_lua_patterns(_PROJECT_DIR)
    print(f'  {len(lua_patterns)} files in patterns/The Green Guide/')

    print('Initialising pattern engine ...', flush=True)
    engine = PatternEngineV3()
    loaded = len(engine.lua_patterns)
    print(f'  {loaded} patterns loaded by engine')

    print('Running verification ...\n', flush=True)
    reports = run_verification(engine, lua_patterns, ods_data)
    print_report(reports, examples_only=examples_only, failures_only=failures_only)


if __name__ == '__main__':
    main()
