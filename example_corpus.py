"""Precomputed "random matching serial" corpus for the Pattern Manager.

Runs every loaded pattern over a generated test corpus (random serials plus
structured "fancy" shapes: solids, radars, repeaters, ladders, pairs, limited
digit sets, years, dates, low serials, bookends, counting ladders) and records
which serials each pattern matched. The Pattern Manager then offers a *verified*,
varied example on demand -- including serials that were generated for OTHER
pattern families but which this pattern also happens to match (e.g. a "4 pairs"
pattern inherits every radar/repeater in the pool that is coincidentally four
pairs). That cross-family pooling is why it feels varied instead of cycling the
same three header examples.

Built lazily and cached to the user-data dir, keyed by a signature of the loaded
patterns so it is rebuilt whenever patterns are added / edited / removed (covers
user patterns and the evolving Green Guide). Brute-forcing a random serial until
it matches is instant for common patterns but never terminates for a 1-in-12M
pattern, so the corpus is precomputed: its rare-family serials are *constructed*
to hit those shapes rather than stumbled on.

NOTE: this is distinct from `pattern_corpus.py`, which is the Duplicate Reviewer's
behavioural-similarity corpus (bitmaps + duplicate groups). This module only
serves random verified examples for the Pattern Manager.

Adapted from dollardetective-web scripts/pattern-similarity.py (the corpus half
only -- this desktop build does not compute the duplicate-similarity matrix).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from pathlib import Path
from typing import Callable, Optional

SEED = 20260911
# Bump when build_corpus_serials() changes shape/coverage, so existing caches
# (whose signature is otherwise patterns-only) rebuild with the new corpus.
CORPUS_VERSION = 2


def cache_path() -> Path:
    """Where the built corpus is cached (per user, alongside user_settings.yaml)."""
    from resource_path import user_data_dir
    return user_data_dir() / "example_corpus.json"


# ---- Corpus generation (ported from the web script) --------------------------------

def build_corpus_serials(rng: random.Random) -> list:
    """The generated test corpus: random serials plus every fancy family, each as a
    full serial (letter prefix + 8 digits + letter/star suffix)."""
    digits8 = set()
    D = "0123456789"
    r = lambda k: "".join(rng.choice(D) for _ in range(k))  # noqa: E731

    def add(s):
        if len(s) == 8 and s.isdigit():
            digits8.add(s)

    for _ in range(4000):
        add(r(8))
    for d in D:  # solids and near-solids, N of a kind
        add(d * 8)
        for _ in range(25):
            s = list(d * 8)
            for _ in range(rng.choice([1, 1, 2, 3, 4])):
                s[rng.randrange(8)] = rng.choice(D)
            add("".join(s))
    for _ in range(600):  # radars, mini radars, near radars
        h = r(4)
        add(h + h[::-1])
        k = rng.choice([5, 6, 7])
        inner = r(k // 2)
        mid = inner + (r(1) if k % 2 else "") + inner[::-1]
        pos = rng.randrange(0, 8 - k + 1)
        add(r(pos) + mid + r(8 - k - pos))
        s = list(h + h[::-1]); s[rng.randrange(8)] = rng.choice(D); add("".join(s))
    for _ in range(500):  # repeaters and bookends
        a = r(4); add(a + a)
        b = r(2); add(b * 4)
        c = r(3); add(c + r(2) + c)
        add(b + r(4) + b)
    for _ in range(400):  # ladders
        start = rng.randrange(10)
        step = rng.choice([1, -1])
        seq = "".join(str((start + step * i) % 10) for i in range(8))
        add(seq)
        n = rng.choice([4, 5, 6, 7])
        part = seq[:n]; pos = rng.randrange(0, 8 - n + 1)
        add(r(pos) + part + r(8 - n - pos))
        shuffled = list(seq); rng.shuffle(shuffled); add("".join(shuffled))
        add("".join(ch * 2 for ch in seq[:4]))
    for _ in range(500):  # pairs, quads, full houses
        ds = rng.sample(D, 4)
        add("".join(x * 2 for x in ds))
        s = list("".join(x * 2 for x in ds)); rng.shuffle(s); add("".join(s))
        a, b = rng.sample(D, 2)
        fh = a * 3 + b * 2; pos = rng.randrange(0, 4)
        add(r(pos) + fh + r(3 - pos))
        add(a * 4 + r(4)); add(r(4) + a * 4); add(a * 4 + b * 4)
    for alphabet in (["0", "1"], ["0", "1", "6", "8", "9"], ["0", "1", "2", "3", "4"], list("02468"), list("13579")):
        for _ in range(250):
            add("".join(rng.choice(alphabet) for _ in range(8)))
    for _ in range(600):  # 2 and 3 distinct digits
        k = rng.choice([2, 3])
        alphabet = rng.sample(D, k)
        add("".join(rng.choice(alphabet) for _ in range(8)))
    for _ in range(900):  # years and dates in the common layouts
        year = rng.randrange(1776, 2040)
        pos = rng.randrange(0, 5)
        add(r(pos) + str(year) + r(4 - pos))
        day = dt.date(rng.randrange(1900, 2040), 1, 1) + dt.timedelta(days=rng.randrange(365))
        add(day.strftime("%m%d%Y")); add(day.strftime("%d%m%Y")); add(day.strftime("%Y%m%d"))
        add(r(2) + day.strftime("%m%d%y")); add(day.strftime("%m%d%y") + r(2))
    for _ in range(400):  # low and high serials, zeros
        add("0000" + r(4)); add("00000" + r(3)); add("000000" + r(2)); add("0000000" + r(1))
        add(r(4) + "0000"); add(r(5) + "000"); add("9999" + r(4))
    for _ in range(300):  # alternators / skips
        a, b = rng.sample(D, 2)
        add((a + b) * 4); add((a + r(1)) * 4)
    # "counting" ladders: four 2-digit numbers stepping by a constant (Counting By
    # 5s/6s, Alternator Ladder, ...). Fully enumerated -- these patterns have only
    # dozens of valid serials each and none arise by chance, so without this they'd
    # offer only their own header examples.
    for step in range(1, 26):
        for direction in (1, -1):
            s = step * direction
            for start in range(100):
                vals = [start + s * i for i in range(4)]
                if all(0 <= v <= 99 for v in vals):
                    add("".join(f"{v:02d}" for v in vals))
    for _ in range(500):  # chunky ladders: 5 or 6 consecutive digits, sorted, over 8 slots
        run = rng.choice([5, 6])
        base = rng.randrange(0, 10 - run + 1)
        vals = [base + i for i in range(run)]
        counts = [1] * run
        for _ in range(8 - run):
            counts[rng.randrange(run)] += 1
        seq = "".join(str(v) * c for v, c in zip(vals, counts))
        add(seq); add(seq[::-1])
    for a in range(10):  # quad ladder: AAAABBBB with B one higher/lower than A
        for b in (a - 1, a + 1):
            if 0 <= b <= 9:
                add(str(a) * 4 + str(b) * 4)

    letters = "ABCDEFGHIJKL"
    serials = []
    for s in sorted(digits8):
        prefix = rng.choice(letters)
        suffix = "*" if rng.random() < 0.08 else rng.choice(letters)
        serials.append(f"{prefix}{s}{suffix}")
    return serials


# ---- Signature / build / cache -----------------------------------------------------

def signature(engine) -> str:
    """A hash of the loaded patterns; changes when any pattern's script, examples or
    tier changes, or a pattern is added/removed -- so the cache rebuilds on demand."""
    h = hashlib.sha1()
    h.update(f"corpus-v{CORPUS_VERSION}\x01".encode("ascii"))
    for name in sorted(engine.lua_patterns):
        info = engine.lua_patterns[name]
        h.update(name.encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update((info.script or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update(repr(list(info.examples or [])).encode("utf-8", "ignore"))
        h.update(str(getattr(info, "tier", "")).encode("utf-8", "ignore"))
        h.update(b"\x01")
    return h.hexdigest()


def build(engine, progress_cb: Optional[Callable[[int, int], None]] = None,
          cancel: Optional[Callable[[], bool]] = None) -> Optional[dict]:
    """Run every pattern over the corpus and record each pattern's matching serials.

    The given engine should have ALL patterns enabled (classify() skips disabled
    ones) -- callers building in the background use a private engine instance for
    exactly this reason. Returns None if `cancel()` becomes true mid-build.
    """
    rng = random.Random(SEED)
    serials = build_corpus_serials(rng)

    # Each pattern's own header examples join the corpus so even a pattern that no
    # generated shape happens to hit is still represented by something verified.
    for name in sorted(engine.lua_patterns):
        for ex in (engine.lua_patterns[name].examples or []):
            text = str(ex).strip()
            if not text:
                continue
            full = f"A{text}B" if (text.isdigit() and len(text) == 8) else text
            serials.append(full)

    serials = list(dict.fromkeys(serials))  # dedupe, keep order
    total = len(serials)

    hits: dict = {name: [] for name in engine.lua_patterns}
    for i, s in enumerate(serials):
        try:
            for name in engine.classify_simple(s):
                lst = hits.get(name)
                if lst is not None:
                    lst.append(i)
        except Exception:
            pass
        if cancel and (i % 200 == 0) and cancel():
            return None
        if progress_cb and (i % 200 == 0):
            progress_cb(i, total)
    if progress_cb:
        progress_cb(total, total)

    return {
        "signature": signature(engine),
        "seed": SEED,
        "serials": serials,
        # Store only patterns that actually hit something -- keeps the file small
        # and makes "no example" a simple missing-key check.
        "hits": {name: idxs for name, idxs in hits.items() if idxs},
    }


def save(data: dict, path: Optional[Path] = None) -> None:
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def load(engine, path: Optional[Path] = None) -> Optional[dict]:
    """The cached corpus if it exists and matches the current patterns, else None."""
    path = path or cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("signature") != signature(engine):
        return None
    return data


def random_serial(data: Optional[dict], name: str, rng=None) -> Optional[str]:
    """A uniformly random verified serial this pattern matched, or None if the corpus
    recorded no hits for it (e.g. plate/seal/serial-range patterns about the note
    itself rather than its digits)."""
    if not data:
        return None
    idxs = (data.get("hits") or {}).get(name)
    if not idxs:
        return None
    serials = data.get("serials") or []
    i = (rng or random).choice(idxs)
    if 0 <= i < len(serials):
        return serials[i]
    return None
