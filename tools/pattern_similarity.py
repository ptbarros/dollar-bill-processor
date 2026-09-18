#!/usr/bin/env python3
"""Generate the pattern-similarity corpus for the desktop Duplicate Reviewer.

Runs EVERY pattern (enabled or not) over ~20k seeded test serials and writes a compressed
bitmap per pattern to pattern_hits.json in the user data dir. Duplicate groups are
computed on load from those bitmaps (see pattern_corpus.py) — we ship one file, not two.

    python tools/pattern_similarity.py               # write to the default cache path
    python tools/pattern_similarity.py --out x.json  # write elsewhere
    python tools/pattern_similarity.py --jobs 1       # single process (slower, simpler)

Ported from dollardetective-web scripts/pattern-similarity.py. Differences:
- uses engine.execute_pattern(name, serial, META) per pattern (ignores enable-state and
  injects DataFile data) instead of the web's SerialLookup;
- PINS date metadata (pattern_corpus.META) for reproducibility;
- writes only the bitmaps (one file); no essentials fields.
"""
import argparse
import base64
import datetime as dt
import json
import multiprocessing as mp
import random
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_corpus import (  # noqa: E402
    META, SEED, Corpus, default_hits_path, duplicate_groups, patterns_version,
)

# ---- Test corpus (seeded; ported verbatim from the web generator) -------------------

def corpus(rng: random.Random) -> list[str]:
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
    today = dt.date(2026, 9, 11)
    for _ in range(900):  # years and dates in the common layouts
        year = rng.randrange(1776, 2040)
        pos = rng.randrange(0, 5)
        add(r(pos) + str(year) + r(4 - pos))
        day = dt.date(rng.randrange(1900, 2040), 1, 1) + dt.timedelta(days=rng.randrange(365))
        add(day.strftime("%m%d%Y")); add(day.strftime("%d%m%Y")); add(day.strftime("%Y%m%d"))
        add(r(2) + day.strftime("%m%d%y")); add(day.strftime("%m%d%y") + r(2))
    add(today.strftime("%m%d%Y"))
    for _ in range(400):  # low and high serials, zeros
        add("0000" + r(4)); add("00000" + r(3)); add("000000" + r(2)); add("0000000" + r(1))
        add(r(4) + "0000"); add(r(5) + "000"); add("9999" + r(4))
    for _ in range(300):  # alternators / skips
        a, b = rng.sample(D, 2)
        add((a + b) * 4); add((a + r(1)) * 4)

    letters = "ABCDEFGHIJKL"
    serials = []
    for s in sorted(digits8):
        prefix = rng.choice(letters)
        suffix = "*" if rng.random() < 0.08 else rng.choice(letters)
        serials.append(f"{prefix}{s}{suffix}")
    return serials


# ---- Running patterns (multiprocessing, engine per worker) --------------------------

_engine = None
_names = None


def _init():
    global _engine, _names
    from pattern_engine_v3 import PatternEngineV3
    _engine = PatternEngineV3()
    _names = sorted(_engine.lua_patterns)


def _run(chunk):
    out = []
    for full in chunk:
        matched = [n for n in _names if _engine.execute_pattern(n, full, META).matched]
        out.append(matched)
    return out


def build_serials(engine):
    """Seeded corpus + every pattern's own Examples (so rare patterns are represented)."""
    serials = corpus(random.Random(SEED))
    example_serials: dict[str, list[str]] = {}
    for name in sorted(engine.lua_patterns):
        for ex in engine.lua_patterns[name].examples or []:
            text = str(ex).strip()
            if not text:
                continue
            full = f"A{text}B" if text.isdigit() and len(text) == 8 else text
            serials.append(full)
            example_serials.setdefault(name, []).append(full)
    serials = list(dict.fromkeys(serials))  # de-dupe, keep first order
    return serials, example_serials


def run_patterns(serials, jobs):
    started = time.time()
    results: list[list[str]] = []
    if jobs == 1:
        _init()
        for i in range(0, len(serials), 50):
            results.extend(_run(serials[i:i + 50]))
            if (i // 50) % 20 == 0:
                print(f"  {len(results):,}/{len(serials):,} serials ({time.time() - started:.0f}s)", flush=True)
    else:
        chunks = [serials[i:i + 50] for i in range(0, len(serials), 50)]
        with mp.Pool(jobs, initializer=_init) as pool:
            for i, part in enumerate(pool.imap(_run, chunks)):
                results.extend(part)
                if i % 20 == 0:
                    print(f"  {len(results):,}/{len(serials):,} serials ({time.time() - started:.0f}s)", flush=True)
    return results


def write_hits(path, engine, serials, results):
    total = len(serials)
    names = sorted(engine.lua_patterns)
    hit_indexes = {n: [] for n in names}
    for idx, matched in enumerate(results):
        for name in matched:
            hit_indexes[name].append(idx)
    packed = {}
    for name in names:
        bits = 0
        for i in hit_indexes[name]:
            bits |= 1 << i
        raw = bits.to_bytes((total + 7) // 8, "little")
        packed[name] = base64.b64encode(zlib.compress(raw, 9)).decode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "patterns_version": patterns_version(engine),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": {"serials": total, "seed": SEED, "meta": META},
        "serials": serials,
        "matches_per_serial": [len(r) for r in results],
        "hits": packed,
    }) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="output path (default: user data dir)")
    ap.add_argument("--jobs", type=int, default=mp.cpu_count(), help="worker processes")
    args = ap.parse_args()

    from pattern_engine_v3 import PatternEngineV3
    engine = PatternEngineV3()
    out = args.out or default_hits_path()

    serials, _ = build_serials(engine)
    print(f"{len(serials):,} test serials, {len(engine.lua_patterns)} patterns, "
          f"{args.jobs} worker(s)", flush=True)

    results = run_patterns(serials, args.jobs)
    write_hits(out, engine, serials, results)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB)", flush=True)

    # Sanity: load it back and compute groups (the on-load path the GUI will use).
    corp = Corpus(json.loads(out.read_text()))
    info_map = {n: {"library": p.library, "display_name": p.display_name or n}
                for n, p in engine.lua_patterns.items()}
    groups = duplicate_groups(corp, info_map)
    exact = sum(1 for g in groups if g["exact"])
    members = sum(len(g["members"]) for g in groups)
    no_hits = [n for n in sorted(engine.lua_patterns) if not corp.bits(n)]
    print(f"{len(groups)} duplicate groups ({exact} exact), {members} patterns", flush=True)
    print(f"patterns with no test hits: {len(no_hits)} {no_hits}", flush=True)
    for g in groups:
        tag = "exact" if g["exact"] else f"min j={g['min_jaccard']}"
        print(f"  {g['id']} [{tag}] {g['members']}")


if __name__ == "__main__":
    main()
