"""
Render the Markdown user guide (docs/USER_GUIDE.md) to a self-contained, styled
HTML page so it can be opened locally in the browser — no internet, no GitHub.

Keeps USER_GUIDE.md as the single source of truth: this converts the small
Markdown subset the guide uses (headings, bold/italic/code, links, tables,
blockquotes, ordered/unordered lists, horizontal rules) at runtime. Headings get
GitHub-style slug ids so the in-page Contents links work.
"""

import re
import html as _html


def _slug(text: str) -> str:
    """GitHub-style heading slug: lowercase, punctuation dropped, spaces→hyphens."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)   # drop punctuation (incl. * & — etc.)
    s = re.sub(r"\s", "-", s)        # each space → hyphen (don't collapse, matches GitHub)
    return s


def _inline(text: str) -> str:
    """Inline Markdown → HTML for one span of text."""
    t = _html.escape(text, quote=False)
    # inline code first (so its contents aren't further formatted in practice)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    # bold, then italic (avoid an escaped \*)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\\)\*([^*]+)\*", r"<em>\1</em>", t)
    t = t.replace("\\*", "*")
    # links [text](url)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def _is_block_start(line: str) -> bool:
    s = line.strip()
    return (not s
            or re.match(r"^#{1,6}\s", s) is not None
            or re.match(r"^-{3,}$", s) is not None
            or s.startswith(">")
            or re.match(r"^(\d+\.|[-*])\s+", s) is not None
            or "|" in line)


def _render_table(rows: list) -> str:
    """rows: list of raw '| a | b |' lines, incl. the separator as rows[1]."""
    def cells(line):
        line = line.strip().strip("|")
        return [c.strip() for c in line.split("|")]
    header = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    out = ['<table>', '<thead><tr>']
    out += [f"<th>{_inline(c)}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _md_to_body(md: str) -> str:
    lines = md.split("\n")
    n = len(lines)
    i = 0
    out = []
    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        # Horizontal rule
        if re.match(r"^-{3,}$", s):
            out.append("<hr>")
            i += 1
            continue
        # Heading
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            text = m.group(2).strip()
            out.append(f'<h{lvl} id="{_slug(text)}">{_inline(text)}</h{lvl}>')
            i += 1
            continue
        # Table: a header row followed by a |---|--- separator
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", lines[i + 1]):
            block = [lines[i], lines[i + 1]]
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                block.append(lines[i])
                i += 1
            out.append(_render_table(block))
            continue
        # Blockquote (consecutive > lines → one quote, wrapped paragraphs merged)
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]).rstrip())
                i += 1
            text = " ".join(x.strip() for x in buf if x.strip())
            out.append(f"<blockquote>{_inline(text)}</blockquote>")
            continue
        # Lists (ordered / unordered); wrapped continuation lines join the item
        lm = re.match(r"^(\d+\.|[-*])\s+(.*)$", s)
        if lm:
            ordered = bool(re.match(r"^\d+\.", s))
            tag = "ol" if ordered else "ul"
            items = []
            while i < n:
                cur = lines[i]
                cs = cur.strip()
                im = re.match(r"^(\d+\.|[-*])\s+(.*)$", cs)
                if im:
                    items.append(im.group(2).strip())
                    i += 1
                elif cs and not _is_block_start(cur):
                    # continuation of the previous item (wrapped line / indent)
                    if items:
                        items[-1] += " " + cs
                    i += 1
                else:
                    break
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        # Paragraph: gather until a blank line or the next block
        buf = [s]
        i += 1
        while i < n and lines[i].strip() and not _is_block_start(lines[i]):
            buf.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(buf))}</p>")
    return "\n".join(out)


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { max-width: 860px; margin: 0 auto; padding: 32px 20px 96px;
  font: 16px/1.65 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #1b1f24; background: #ffffff; }
h1 { font-size: 30px; border-bottom: 2px solid #e6e8eb; padding-bottom: .3em; }
h2 { font-size: 23px; margin-top: 1.8em; border-bottom: 1px solid #e6e8eb; padding-bottom: .3em; }
h3 { font-size: 18px; margin-top: 1.5em; }
h4 { font-size: 16px; margin-top: 1.3em; }
a { color: #1a66c2; text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: #f0f2f4; padding: .12em .35em; border-radius: 4px;
  font: 13.5px "SF Mono", Consolas, "Liberation Mono", monospace; }
blockquote { margin: 1em 0; padding: .5em 1em; color: #3a4149;
  background: #f7f9fb; border-left: 4px solid #c7ced6; border-radius: 0 6px 6px 0; }
blockquote strong { color: #1b1f24; }
table { border-collapse: collapse; width: 100%; margin: 1.1em 0; font-size: 14.5px; }
th, td { border: 1px solid #d6dbe1; padding: 8px 11px; text-align: left; vertical-align: top; }
th { background: #f0f2f4; }
tr:nth-child(even) td { background: #fafbfc; }
hr { border: 0; border-top: 1px solid #e6e8eb; margin: 2em 0; }
ul, ol { padding-left: 1.5em; }
li { margin: .3em 0; }
"""


def render_guide_html(md_text: str, title: str = "Dollar Detective — User Guide") -> str:
    """Return a full standalone HTML document for the given Markdown guide text."""
    body = _md_to_body(md_text)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body>\n{body}\n</body></html>"
    )
