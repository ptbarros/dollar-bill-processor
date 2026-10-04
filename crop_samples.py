"""
Built-in sample bills for the crop tool.

The crop tool ships one front+back scan per denomination under
``assets/samples/<denom>/`` so a new user can try cropping immediately, without
first pointing the tool at a folder of their own. Discovery is data-driven: each
sub-folder that holds a front image and its ``_b`` back is offered as a sample, so
dropping in a new ``assets/samples/twenty_dollar/`` (with ``twenty_dollar.jpg`` +
``twenty_dollar_b.jpg``) adds that denomination with no code change.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from resource_path import app_base

# Nice labels + display order for the denominations we know. Unknown folder names
# fall back to a title-cased label and sort last (alphabetically).
_LABELS = {
    'one_dollar': '$1', 'two_dollar': '$2', 'five_dollar': '$5',
    'ten_dollar': '$10', 'twenty_dollar': '$20', 'fifty_dollar': '$50',
    'hundred_dollar': '$100',
}
_ORDER = list(_LABELS.keys())

_IMG_EXTS = ('.jpg', '.jpeg', '.png')


def samples_root() -> Path:
    return app_base() / 'assets' / 'samples'


def pretty_label(key: str) -> str:
    if key in _LABELS:
        return _LABELS[key]
    return key.replace('_', ' ').title()


def _find_front_back(folder: Path):
    """Return (front, back) paths for a sample folder, or (None, None).

    Front = an image whose stem does NOT end in ``_b``; back = its ``_b`` sibling.
    """
    imgs = [p for p in sorted(folder.iterdir())
            if p.suffix.lower() in _IMG_EXTS] if folder.is_dir() else []
    fronts = [p for p in imgs if not p.stem.lower().endswith('_b')]
    if not fronts:
        return None, None
    front = fronts[0]
    back = None
    for p in imgs:
        if p.stem.lower() == front.stem.lower() + '_b':
            back = p
            break
    return front, back


def discover() -> List[dict]:
    """List available samples as dicts: {key, label, dir, front, back}.

    Ordered by the known-denomination order, then any extras alphabetically.
    """
    root = samples_root()
    if not root.is_dir():
        return []
    found = {}
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        front, back = _find_front_back(sub)
        if front is None:
            continue
        found[sub.name] = {
            'key': sub.name,
            'label': pretty_label(sub.name),
            'dir': sub,
            'front': front,
            'back': back,
        }
    ordered = [found[k] for k in _ORDER if k in found]
    extras = sorted((v for k, v in found.items() if k not in _ORDER),
                    key=lambda d: d['label'])
    return ordered + extras


def has_samples() -> bool:
    return len(discover()) > 0


def get(key: str) -> Optional[dict]:
    for s in discover():
        if s['key'] == key:
            return s
    return None
