#!/usr/bin/env python3
"""
review_logic.py - Generate a compact logic review report comparing each
implemented Green Guide Lua pattern's logic against the ODS book description.

For each pattern outputs:
  - CS# and display name
  - ODS description (col 8, book prose)
  - Lua Description: header field (abbreviated)
  - match() function body (comments and blank lines stripped)

This produces a pre-digested report for human review without reading raw
files one at a time in a conversation context.

Usage:
    python3 tools/review_logic.py                      # all 123 patterns
    python3 tools/review_logic.py --filter CS-10,CS-20,CS-40
    python3 tools/review_logic.py --batch 7            # batch 7 patterns
    python3 tools/review_logic.py -o review.txt        # write to file

Batch definitions:
    7 = CS-10,20,40,80,90,140,180,220,300,320,340,350,390,450,970,1030,1080,1500,1510
"""

import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# ODS constants
# ---------------------------------------------------------------------------
ODS_PATH  = Path.home() / 'projects' / 'tggfsn.ods'

COL_SKIP     = 3
COL_CREATED  = 4
COL_CS       = 5
COL_NAME     = 6
COL_EXAMPLES = 7
COL_DESC     = 8

NS_TABLE = 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'
NS_TEXT  = 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'

BATCHES = {
    7: [10, 20, 40, 80, 90, 140, 180, 220, 300, 320, 340, 350, 390, 450,
        970, 1030, 1080, 1500, 1510],
}

# ---------------------------------------------------------------------------
# ODS helpers (shared with verify_patterns.py)
# ---------------------------------------------------------------------------

def _get_cell_text(cell):
    parts = [p.text for p in cell.findall(f'.//{{{NS_TEXT}}}p') if p.text]
    return ' | '.join(parts) if parts else ''


def _get_row_cells(row, n_cols):
    cols = []
    for cell in row.findall(f'{{{NS_TABLE}}}table-cell'):
        rep   = cell.get(f'{{{NS_TABLE}}}number-columns-repeated')
        count = int(rep) if rep else 1
        text  = _get_cell_text(cell)
        cols.extend([text] * count)
        if len(cols) >= n_cols:
            break
    return (cols + [''] * n_cols)[:n_cols]


def load_ods_all(ods_path):
    """Return dict keyed by 'CS-NNN' for ALL rows (implemented or not)."""
    with zipfile.ZipFile(ods_path) as z:
        content = z.read('content.xml').decode('utf-8')
    root  = ET.fromstring(content)
    ns    = {'table': NS_TABLE}
    sheet = root.findall('.//table:table', ns)[0]
    rows  = sheet.findall('table:table-row', ns)[1:]  # skip header

    result = {}
    for row in rows:
        cells  = _get_row_cells(row, 10)
        cs_num = cells[COL_CS].strip()
        if not cs_num:
            continue
        cs_key = f'CS-{cs_num}'
        result[cs_key] = {
            'cs_key':   cs_key,
            'name':     cells[COL_NAME].strip(),
            'desc':     cells[COL_DESC].strip(),
            'skip':     cells[COL_SKIP].strip(),
            'created':  cells[COL_CREATED].strip(),
        }
    return result


# ---------------------------------------------------------------------------
# Lua helpers
# ---------------------------------------------------------------------------

def parse_lua_header(script):
    meta = {}
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


def extract_match_body(script):
    """
    Extract the match() function body, stripped of:
      - pure comment lines  (lines where only content is -- ...)
      - blank lines
    Returns the condensed lines as a single string.
    """
    # Find function match(ctx) ... end
    m = re.search(r'function\s+match\s*\([^)]*\)(.*?)^end\b',
                  script, re.DOTALL | re.MULTILINE)
    if not m:
        return '(match() not found)'

    body = m.group(1)
    lines = body.split('\n')
    kept  = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue                    # blank
        if stripped.startswith('--'):
            continue                    # pure comment line
        # Inline comments: keep the code part, strip the comment
        code_part = re.sub(r'\s*--[^"\']*$', '', line).rstrip()
        if code_part.strip():
            kept.append(code_part)

    return '\n'.join(kept)


def load_lua_patterns(project_dir):
    gg_dir   = project_dir / 'patterns' / 'The Green Guide'
    results  = []
    for lua_file in sorted(gg_dir.glob('*.lua')):
        script = lua_file.read_text(encoding='utf-8')
        meta   = parse_lua_header(script)
        if not meta:
            continue
        body = extract_match_body(script)
        results.append({
            'file':        lua_file,
            'pattern':     meta.get('pattern', lua_file.stem.upper()),
            'displayname': meta.get('displayname', ''),
            'description': meta.get('description', ''),
            'bookref':     meta.get('bookref', ''),
            'body':        body,
        })
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

SEP  = '=' * 72
SEP2 = '-' * 72

def wrap(text, width=68, indent='  '):
    if not text:
        return f'{indent}(none)'
    words  = text.split()
    lines  = []
    cur    = indent
    for w in words:
        if len(cur) + len(w) + 1 > width and cur != indent:
            lines.append(cur.rstrip())
            cur = indent + w + ' '
        else:
            cur += w + ' '
    if cur.strip():
        lines.append(cur.rstrip())
    return '\n'.join(lines)


def generate_report(lua_patterns, ods_data, filter_cs=None, out=None):
    if out is None:
        out = sys.stdout

    if filter_cs:
        filter_set = {f'CS-{n}' for n in filter_cs}
        lua_patterns = [p for p in lua_patterns if p['bookref'] in filter_set]

    total = len(lua_patterns)
    out.write(f'Logic Review Report — {total} patterns\n')
    out.write(f'ODS: {ODS_PATH}\n')
    out.write(SEP + '\n\n')

    for i, lua in enumerate(lua_patterns, 1):
        bookref = lua['bookref']
        ods     = ods_data.get(bookref, {})

        out.write(f'[{i}/{total}]  {bookref}  —  {lua["displayname"]}\n')
        out.write(f'File: {lua["file"].name}\n')
        out.write(SEP2 + '\n')

        out.write('ODS DESCRIPTION:\n')
        out.write(wrap(ods.get('desc', '(not found in ODS)')) + '\n\n')

        out.write('LUA DESCRIPTION:\n')
        out.write(wrap(lua['description']) + '\n\n')

        out.write('MATCH() LOGIC:\n')
        for line in lua['body'].split('\n'):
            out.write(f'  {line}\n')

        out.write('\n' + SEP + '\n\n')

    out.write(f'END OF REPORT — {total} patterns reviewed\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args       = sys.argv[1:]
    filter_cs  = None
    out_path   = None
    batch_num  = None

    i = 0
    while i < len(args):
        if args[i] == '--filter' and i + 1 < len(args):
            # --filter CS-10,CS-20,CS-40  OR  10,20,40
            raw = args[i + 1].replace('CS-', '').split(',')
            filter_cs = [r.strip() for r in raw if r.strip()]
            i += 2
        elif args[i] == '--batch' and i + 1 < len(args):
            batch_num = int(args[i + 1])
            if batch_num not in BATCHES:
                print(f'Unknown batch {batch_num}. Defined batches: {list(BATCHES)}')
                sys.exit(1)
            filter_cs = [str(n) for n in BATCHES[batch_num]]
            i += 2
        elif args[i] == '-o' and i + 1 < len(args):
            out_path = Path(args[i + 1])
            i += 2
        else:
            i += 1

    print('Loading ODS ...', flush=True)
    ods_data = load_ods_all(ODS_PATH)
    print(f'  {len(ods_data)} rows loaded')

    print('Loading Lua patterns ...', flush=True)
    lua_patterns = load_lua_patterns(_PROJECT_DIR)
    print(f'  {len(lua_patterns)} files found')

    if batch_num:
        print(f'Filtering to batch {batch_num} ({len(filter_cs)} patterns)')
    elif filter_cs:
        print(f'Filtering to {len(filter_cs)} specified patterns')

    if out_path:
        print(f'Writing report to {out_path} ...', flush=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            generate_report(lua_patterns, ods_data, filter_cs=filter_cs, out=f)
        print('Done.')
    else:
        generate_report(lua_patterns, ods_data, filter_cs=filter_cs)


if __name__ == '__main__':
    main()
