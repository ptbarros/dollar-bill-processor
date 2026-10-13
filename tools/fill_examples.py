#!/usr/bin/env python3
"""
fill_examples.py - Populate Description and Examples columns in tggfsn.ods
from the Green Guide OCR text file (~/Pictures/tggfsn.txt).

Usage:
    python3 tools/fill_examples.py [--dry-run] [--chapter N]

    --dry-run    Print proposed changes without writing anything.
    --chapter N  Only process rows whose Chapter column equals N.
                 Omit to process all chapters.
"""

import re
import sys
import zipfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from copy import deepcopy

# File paths
ODS_PATH = Path.home() / 'projects' / 'tggfsn.ods'
TXT_PATH = Path.home() / 'Pictures' / 'tggfsn.txt'

# ODS namespace URIs
NS_TABLE  = 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'
NS_TEXT   = 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'
NS_OFFICE = 'urn:oasis:names:tc:opendocument:xmlns:office:1.0'

# Column indices (0-based)
# WARNING: These constants are STALE as of 2026-02-23. The ODS column layout changed:
#   Col 5 = CS-#, Col 6 = Name, Col 7 = Examples, Col 8 = Description
# Update these constants before reusing this script.
COL_CHAPTER  = 1
COL_NAME     = 5
COL_DESC     = 6
COL_CS       = 7
COL_EXAMPLES = 8

DRY_RUN = '--dry-run' in sys.argv

# Parse --chapter N argument (None = all chapters)
_CHAPTER_FILTER: int | None = None
for _i, _arg in enumerate(sys.argv):
    if _arg == '--chapter' and _i + 1 < len(sys.argv):
        _CHAPTER_FILTER = int(sys.argv[_i + 1])


# ---------------------------------------------------------------------------
# ODS helpers
# ---------------------------------------------------------------------------

def register_namespaces_from_content(content: str) -> None:
    """Register all namespace prefixes found in the XML header."""
    for prefix, uri in re.findall(r'xmlns:([^=\s]+)\s*=\s*"([^"]+)"', content[:6000]):
        try:
            ET.register_namespace(prefix, uri)
        except ValueError:
            pass


def get_ns_calcext(content: str) -> str | None:
    m = re.search(r'xmlns:calcext\s*=\s*"([^"]+)"', content[:6000])
    return m.group(1) if m else None


def get_cell_text(cell: ET.Element) -> str:
    parts = [p.text for p in cell.findall(f'.//{{{NS_TEXT}}}p') if p.text]
    return ' | '.join(parts) if parts else ''


def get_row_value(row: ET.Element, target_col: int) -> str:
    """Get cell value at target_col, expanding number-columns-repeated."""
    col = 0
    for cell in row.findall(f'{{{NS_TABLE}}}table-cell'):
        rep = cell.get(f'{{{NS_TABLE}}}number-columns-repeated')
        n = int(rep) if rep else 1
        if col <= target_col < col + n:
            return get_cell_text(cell)
        col += n
    return ''


def _set_string_value(cell: ET.Element, value: str, ns_calcext: str | None) -> None:
    """Fill (or overwrite) a cell element with a string value."""
    for child in list(cell):
        cell.remove(child)
    # Remove repeated attribute
    rep_key = f'{{{NS_TABLE}}}number-columns-repeated'
    cell.attrib.pop(rep_key, None)
    # Set value-type attributes
    cell.set(f'{{{NS_OFFICE}}}value-type', 'string')
    if ns_calcext:
        cell.set(f'{{{ns_calcext}}}value-type', 'string')
    p = ET.SubElement(cell, f'{{{NS_TEXT}}}p')
    p.text = value


def set_cell_at_col(row: ET.Element, target_col: int, value: str,
                    ns_calcext: str | None) -> None:
    """Set cell at target_col, splitting repeated-cell groups if needed."""
    col = 0
    cells = list(row.findall(f'{{{NS_TABLE}}}table-cell'))

    for ci, cell in enumerate(cells):
        rep = cell.get(f'{{{NS_TABLE}}}number-columns-repeated')
        n = int(rep) if rep else 1

        if col <= target_col < col + n:
            if n == 1:
                _set_string_value(cell, value, ns_calcext)
            else:
                # Split the repeated-cell group around target_col
                before = target_col - col
                after  = col + n - target_col - 1

                # Find insertion position in row children
                row.remove(cell)
                # Count past already-consumed columns to find insert index
                insert_pos = 0
                c = 0
                for i, ch in enumerate(list(row)):
                    if ch.tag == f'{{{NS_TABLE}}}table-cell':
                        r = ch.get(f'{{{NS_TABLE}}}number-columns-repeated')
                        c += int(r) if r else 1
                    if c >= col:
                        insert_pos = i + 1
                        break

                if before > 0:
                    bc = deepcopy(cell)
                    for k in list(bc.attrib):
                        if 'value-type' in k or (f'{{{NS_OFFICE}}}value' == k):
                            del bc.attrib[k]
                    for ch in list(bc):
                        bc.remove(ch)
                    bc.set(f'{{{NS_TABLE}}}number-columns-repeated', str(before))
                    row.insert(insert_pos, bc)
                    insert_pos += 1

                new_cell = ET.Element(f'{{{NS_TABLE}}}table-cell')
                _set_string_value(new_cell, value, ns_calcext)
                row.insert(insert_pos, new_cell)
                insert_pos += 1

                if after > 0:
                    ac = deepcopy(cell)
                    for k in list(ac.attrib):
                        if 'value-type' in k or (f'{{{NS_OFFICE}}}value' == k):
                            del ac.attrib[k]
                    for ch in list(ac):
                        ac.remove(ch)
                    ac.set(f'{{{NS_TABLE}}}number-columns-repeated', str(after))
                    row.insert(insert_pos, ac)
            return

        col += n

    print(f"  WARNING: col {target_col} not found in row (row has {col} cols)")


# ---------------------------------------------------------------------------
# M-marker stripping
# ---------------------------------------------------------------------------

def strip_m_markers(examples_str: str) -> str:
    """Convert 'M 12345678 M, M 87654321 M' → '12345678, 87654321'."""
    serials = re.findall(r'M\s+(\d{8})\s+M', examples_str)
    if serials:
        return ', '.join(serials)
    # Already clean — just tidy trailing comma/space
    return examples_str.strip().rstrip(',').strip()


# ---------------------------------------------------------------------------
# Text-file parsing
# ---------------------------------------------------------------------------

def load_text_file(path: Path) -> list[str]:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return [line.rstrip('\n') for line in f]


# Pattern name heading: standalone line "CS-Something." (period optional)
# Allows digits after CS- (e.g. "CS-20AKs"), dots in name (e.g. "CS-E.U.")
_HEADING_RE = re.compile(r'^(CS-[A-Za-z0-9][A-Za-z0-9 ./-]{0,60})\.?\s*$')


def _looks_like_heading(lines: list[str], i: int) -> bool:
    """
    Return True if line i is plausibly a real section heading — i.e. within
    the next 5 lines there is at least one line that looks like prose (longer
    than 20 chars) OR an @CS~ tag.
    """
    for j in range(i + 1, min(len(lines), i + 6)):
        s = lines[j].strip()
        if re.search(r'[@a-zA-Z]?CS[~-]\d', s):
            return True
        if len(s) > 20 and not re.match(r'^\d[\d,]*$', s):
            return True
    return False


def build_text_index(lines: list[str]) -> list[tuple[int, str]]:
    """Return sorted list of (line_idx, name) for all pattern headings."""
    entries = []
    seen: set[int] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = _HEADING_RE.match(stripped)
        if m and i not in seen and _looks_like_heading(lines, i):
            name = re.sub(r'\s+', ' ', m.group(1).strip())
            entries.append((i, name))
            seen.add(i)
    return entries


def _norm(s: str) -> str:
    """Normalise a pattern name for fuzzy comparison."""
    s = s.lower().strip().rstrip('.')
    s = re.sub(r'\s+', ' ', s)
    # Treat OCR 0↔O confusion in "xOAK" names
    s = re.sub(r'(?<=[1-9])0(aks?\b)', r'o\1', s)   # "20aks" → "2oaks"
    # lowercase o already handled above; ensure "oak" stays "oak"
    return s


def find_text_section(lines: list[str], text_index: list[tuple[int, str]],
                      pattern_name: str) -> tuple[int, int] | None:
    """
    Find (start_line, end_line) for pattern_name in the text.
    Returns None if not found.
    """
    target = _norm(pattern_name)

    match_pos = None

    # Pass 1: exact normalised match
    for i, (line_idx, name) in enumerate(text_index):
        if _norm(name) == target:
            match_pos = i
            break

    # Pass 2: strip leading "CS-" from both and compare
    if match_pos is None:
        t2 = re.sub(r'^cs-', '', target)
        for i, (line_idx, name) in enumerate(text_index):
            n2 = re.sub(r'^cs-', '', _norm(name))
            if n2 == t2:
                match_pos = i
                break

    # Pass 3: strip "random " from both and compare
    if match_pos is None:
        t3 = re.sub(r'\brandom\s+', '', target)
        for i, (line_idx, name) in enumerate(text_index):
            n3 = re.sub(r'\brandom\s+', '', _norm(name))
            if n3 == t3 and n3 != _norm(name):   # only if stripping made a diff
                match_pos = i
                break

    if match_pos is None:
        return None

    start_line = text_index[match_pos][0]
    end_line = (text_index[match_pos + 1][0]
                if match_pos + 1 < len(text_index)
                else len(lines))
    return start_line, end_line


# Lines that signal end of the prose description
_STOP_RE = [
    re.compile(r'#\d+\s+M[\s\d]', re.IGNORECASE),          # numbered example
    re.compile(r'^\s*Example\s*$', re.IGNORECASE),
    re.compile(r'^\s*Ex\s*a\s*mple\s*$', re.IGNORECASE),   # OCR artifact
    re.compile(r'^\s*Ex\s*$', re.IGNORECASE),               # OCR artifact "Ex"
    re.compile(r'^\s*Recurrence\s*$', re.IGNORECASE),
    re.compile(r'^\s*Rarity\s*$', re.IGNORECASE),
    re.compile(r'^\s*\$'),                                   # price line
]

# Lines that are OCR noise and should be skipped during description extraction
_SKIP_RE = [
    re.compile(r'^\s*$'),
    re.compile(r'^\s*[@a-zA-Z]?CS[~-]\d'),                  # @CS~NNN or eCS~NNN tags
    re.compile(r'^\s*&'),                                    # chapter heading noise
    re.compile(r'^\s*Currency[~\s]Systems\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d{1,3}\s*$'),                         # page numbers
    re.compile(r'^\s*\d[\d,]+\s*$'),                        # stat numbers
    re.compile(r'^\s*[A-Z]{1,6}\s*$'),                      # ALL-CAPS shorts
    re.compile(r'^[\\"/\'~\-*&<>]{1,4}\s*$'),               # symbol lines
    re.compile(r'^\s*[OoCc]\s*$'),                          # single-letter OCR
    re.compile(r'^\s*[a-zA-Z]{1,4}\s*$'),                   # short word-like junk (any case)
]


def extract_description(lines: list[str], start: int, end: int) -> str | None:
    """Extract prose description from the pattern section."""
    parts = []
    for i in range(start + 1, min(end, start + 80)):
        stripped = lines[i].strip()
        if any(p.search(stripped) for p in _STOP_RE):
            break
        if any(p.match(stripped) for p in _SKIP_RE):
            continue
        parts.append(stripped)

    if not parts:
        return None

    desc = re.sub(r'\s+', ' ', ' '.join(parts)).strip()
    if desc.endswith('.'):
        desc = desc[:-1]
    return desc if len(desc) >= 20 else None


def extract_examples(lines: list[str], start: int, end: int) -> list[str]:
    """Return 8-digit serial strings from '#N M 12345678 M' lines."""
    serial_re = re.compile(r'#\d+\s+M\s*(\d{8})\s*M', re.IGNORECASE)
    serials = []
    for i in range(start, min(end, start + 200)):
        m = serial_re.search(lines[i])
        if m:
            serials.append(m.group(1))
    return serials


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tag = '[DRY RUN] ' if DRY_RUN else ''
    print(f"{tag}Loading files...")

    lines = load_text_file(TXT_PATH)
    print(f"  Text file: {len(lines)} lines")

    with zipfile.ZipFile(ODS_PATH) as z:
        content_xml = z.read('content.xml').decode('utf-8')
        other_files = {name: z.read(name)
                       for name in z.namelist() if name != 'content.xml'}

    register_namespaces_from_content(content_xml)
    ns_calcext = get_ns_calcext(content_xml)

    root = ET.fromstring(content_xml)
    ns = {'table': NS_TABLE, 'text': NS_TEXT, 'office': NS_OFFICE}
    sheet = root.findall('.//table:table', ns)[0]
    data_rows = sheet.findall('table:table-row', ns)[1:]   # skip header
    print(f"  ODS data rows: {len(data_rows)}")

    if _CHAPTER_FILTER is not None:
        data_rows = [r for r in data_rows
                     if get_row_value(r, COL_CHAPTER) == str(_CHAPTER_FILTER)]
        print(f"  Filtered to chapter {_CHAPTER_FILTER}: {len(data_rows)} rows")

    text_index = build_text_index(lines)
    print(f"  Text index entries: {len(text_index)}")

    # Counters
    desc_filled   = 0
    desc_skipped  = 0
    desc_no_text  = 0
    desc_no_match = 0
    ex_filled     = 0
    ex_stripped   = 0
    no_match      = 0

    for row_idx, row in enumerate(data_rows):
        name = get_row_value(row, COL_NAME)
        if not name:
            continue

        cur_desc = get_row_value(row, COL_DESC)
        cur_ex   = get_row_value(row, COL_EXAMPLES)

        # --- Strip M markers from existing examples ---
        if cur_ex and 'M ' in cur_ex:
            clean = strip_m_markers(cur_ex)
            if clean != cur_ex:
                if DRY_RUN:
                    print(f"  STRIP  {name}: {repr(cur_ex)} → {repr(clean)}")
                else:
                    set_cell_at_col(row, COL_EXAMPLES, clean, ns_calcext)
                ex_stripped += 1
                cur_ex = clean  # update for logic below

        # --- Description already filled → skip ---
        if cur_desc:
            desc_skipped += 1

        # --- Find text section ---
        section = find_text_section(lines, text_index, name)
        if section is None:
            no_match += 1
            if DRY_RUN:
                print(f"  NO MATCH: {name}")
            continue

        start, end = section

        # --- Fill description ---
        if not cur_desc:
            new_desc = extract_description(lines, start, end)
            if new_desc:
                if DRY_RUN:
                    print(f"  DESC   {name}: {repr(new_desc[:100])}")
                else:
                    set_cell_at_col(row, COL_DESC, new_desc, ns_calcext)
                desc_filled += 1
            else:
                desc_no_text += 1
                if DRY_RUN:
                    print(f"  NO_TXT {name}")

        # --- Fill examples (only if currently empty) ---
        if not cur_ex:
            serials = extract_examples(lines, start, end)
            if serials:
                ex_str = ', '.join(serials)
                if DRY_RUN:
                    print(f"  EXMPL  {name}: {ex_str}")
                else:
                    set_cell_at_col(row, COL_EXAMPLES, ex_str, ns_calcext)
                ex_filled += 1

    print(f"\nSummary:")
    print(f"  Descriptions filled (new):       {desc_filled}")
    print(f"  Descriptions skipped (existing): {desc_skipped}")
    print(f"  Descriptions: no text found:     {desc_no_text}")
    print(f"  Examples filled (new):           {ex_filled}")
    print(f"  Examples: M-markers stripped:    {ex_stripped}")
    print(f"  Rows with no text match:         {no_match}")

    if DRY_RUN:
        print("\n[DRY RUN] No changes written.")
        return

    # Serialize and save
    print(f"\nWriting {ODS_PATH} ...")

    new_content = ET.tostring(root, encoding='unicode', xml_declaration=False)
    new_content = '<?xml version="1.0" encoding="UTF-8"?>' + new_content

    backup = ODS_PATH.with_suffix('.ods.bak')
    shutil.copy2(ODS_PATH, backup)
    print(f"  Backup → {backup}")

    tmp = ODS_PATH.with_suffix('.ods.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        # mimetype must be first and uncompressed
        if b'mimetype' in other_files.get('mimetype', b''):
            z.writestr(
                zipfile.ZipInfo('mimetype'),
                other_files.pop('mimetype').decode('utf-8')
            )
        elif 'mimetype' in other_files:
            info = zipfile.ZipInfo('mimetype')
            info.compress_type = zipfile.ZIP_STORED
            z.writestr(info, other_files.pop('mimetype').decode('utf-8'))
        for fname, data in other_files.items():
            z.writestr(fname, data)
        z.writestr('content.xml', new_content.encode('utf-8'))

    tmp.replace(ODS_PATH)
    print(f"  Saved  → {ODS_PATH}")


if __name__ == '__main__':
    main()
