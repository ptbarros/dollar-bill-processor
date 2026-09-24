"""Behavioural comparison of patterns for the Duplicate Reviewer.

`tools/pattern_similarity.py` runs every pattern over a fixed, seeded set of ~20k test
serials and writes `pattern_hits.json`: the serial list, how many patterns matched each
serial, and a compressed bitmap per pattern. Comparing two patterns is then a couple of
integer operations — which serials both match, which only one matches, and telling
examples of each (the serials the fewest OTHER patterns fire on).

Duplicate groups are computed here on load from those bitmaps (`duplicate_groups`), so we
ship one file, not two. Patterns that behave identically on every test serial may still
differ; `separate()` keeps looking beyond the corpus by running the two patterns on
serials built around their own examples.

Ported from dollardetective-web `api/app/corpus.py` + the grouping half of its
`scripts/pattern-similarity.py`, with two desktop changes:
- the engine primitive is `engine.execute_pattern(name, serial, META)` (runs one pattern
  regardless of enable-state and injects its DataFile data);
- date metadata is PINNED (META) so the corpus is reproducible — 16 Green Guide date/year
  patterns read current_year/month/day, which would otherwise track "today".
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import time
import zlib
from pathlib import Path

# Seed for the generated test corpus. Changing it invalidates every cached corpus.
SEED = 20260911

# PINNED date metadata for reproducibility. create_context() uses setdefault(), so these
# override date.today(). Must be identical in the generator and in separate(), and is
# folded into patterns_version() so a cache built under a different date is detected.
META = {"current_year": 2026, "current_month": 9, "current_day": 11}

# Two patterns are duplicates when their matches are at least this alike (1.0 = identical
# on the corpus). Duplicates chain into groups via union-find.
DUP_MIN_JACCARD = 0.9

# Order a group's members are shown in: the original libraries, canonical first.
LIBRARY_ORDER = ["core", "Nicks", "user"]

EXAMPLES = 3
GROUP_SERIALS = 5
SEARCH_SECONDS = 5.0
SEARCH_TRIES = 60000


def default_hits_path() -> Path:
    """Where the generated corpus lives — the writable user data dir, not the repo."""
    try:
        from resource_path import user_data_dir
        return user_data_dir() / "pattern_hits.json"
    except Exception:
        return Path(__file__).resolve().parent / "pattern_hits.json"


def patterns_version(engine) -> str:
    """A hash of every pattern's name + script + the pinned metadata. When any pattern is
    added, removed, or edited (or META changes) this changes, so a stale corpus is
    detectable. Enable/disable state is deliberately NOT included — the corpus covers all
    patterns regardless."""
    h = hashlib.sha256()
    for name in sorted(engine.lua_patterns):
        info = engine.lua_patterns[name]
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update((info.script or "").encode("utf-8"))
        h.update(b"\0")
    h.update(json.dumps(META, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


class Corpus:
    def __init__(self, data: dict):
        self.serials: list[str] = data["serials"]
        self.matches_per_serial: list[int] = data["matches_per_serial"]
        self.patterns_version: str = data.get("patterns_version", "")
        self.size: int = len(self.serials)
        self._packed: dict[str, str] = data["hits"]
        self._bits: dict[str, int] = {}
        self._index = {s: i for i, s in enumerate(self.serials)}

    @property
    def names(self) -> list[str]:
        return list(self._packed.keys())

    def has(self, name: str) -> bool:
        return name in self._packed

    def bits(self, name: str) -> int:
        if name not in self._bits:
            self._bits[name] = int.from_bytes(
                zlib.decompress(base64.b64decode(self._packed[name])), "little")
        return self._bits[name]

    def _indexes(self, bits: int) -> list[int]:
        out = []
        while bits:
            low = bits & -bits
            out.append(low.bit_length() - 1)
            bits ^= low
        return out

    def examples(self, bits: int, how_many: int = EXAMPLES) -> list[str]:
        """The most telling serials in a set: fewest other patterns matched them."""
        indexes = self._indexes(bits)
        indexes.sort(key=lambda i: (self.matches_per_serial[i], i))
        return [self.serials[i] for i in indexes[:how_many]]

    def serials_for(self, name: str, how_many: int = 5) -> list[str]:
        return self.examples(self.bits(name), how_many) if self.has(name) else []

    def compare(self, a: str, b: str) -> dict:
        """How two patterns behaved on the corpus, from both sides."""
        A, B = self.bits(a), self.bits(b)
        both, only_a, only_b = A & B, A & ~B, B & ~A
        union = (A | B).bit_count()
        return {
            "a": a, "b": b, "tested": self.size,
            "hits_a": A.bit_count(), "hits_b": B.bit_count(),
            "both": both.bit_count(), "only_a": only_a.bit_count(), "only_b": only_b.bit_count(),
            "jaccard": round(both.bit_count() / union, 3) if union else 0.0,
            "both_serials": self.examples(both),
            "only_a_serials": self.examples(only_a),
            "only_b_serials": self.examples(only_b),
        }


def load(path: Path = None) -> Corpus | None:
    """Read the corpus. Returns None if it does not exist or is unreadable."""
    try:
        return Corpus(json.loads((path or default_hits_path()).read_text()))
    except (OSError, ValueError, KeyError):
        return None


def is_stale(corpus: Corpus, engine) -> bool:
    """True if the corpus was built for a different pattern set than the engine holds."""
    return corpus.patterns_version != patterns_version(engine)


# ---- Duplicate groups (computed on load from the bitmaps) ---------------------------

def duplicate_groups(corpus: Corpus, info_map: dict | None = None) -> list[dict]:
    """Patterns that match (nearly) the same serials, chained into groups via union-find.

    info_map: optional {name: {"library": str, "display_name": str}} used only to order a
    group's members (core -> Nicks -> The Green Guide -> user, canonical first) and to
    sort the group list. Falls back to corpus order / the name when absent.
    """
    info_map = info_map or {}
    names = [n for n in corpus.names if corpus.bits(n)]  # skip patterns with no hits
    bits = {n: corpus.bits(n) for n in names}
    counts = {n: bits[n].bit_count() for n in names}

    pairs = []
    for i, a in enumerate(names):
        A = bits[a]
        for b in names[i + 1:]:
            common = (A & bits[b]).bit_count()
            if not common:
                continue
            union = counts[a] + counts[b] - common
            if union and common / union >= DUP_MIN_JACCARD:
                pairs.append((a, b))

    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)

    members: dict[str, list[str]] = {}
    for n in names:
        members.setdefault(find(n), []).append(n)

    def rank(n):
        info = info_map.get(n, {})
        library = info.get("library", "")
        order = LIBRARY_ORDER.index(library) if library in LIBRARY_ORDER else len(LIBRARY_ORDER)
        return (order, info.get("display_name") or n)

    groups = []
    for group in members.values():
        if len(group) < 2:
            continue
        group.sort(key=rank)
        common_bits = bits[group[0]]
        for n in group[1:]:
            common_bits &= bits[n]
        chosen = corpus.examples(common_bits, GROUP_SERIALS) if common_bits else []
        inside = []
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                common = (bits[a] & bits[b]).bit_count()
                union = counts[a] + counts[b] - common
                inside.append({
                    "a": a, "b": b,
                    "jaccard": round(common / union, 3) if union else 0.0,
                    "both": common,
                    "only_a": counts[a] - common, "only_b": counts[b] - common,
                    "only_a_serials": corpus.examples(bits[a] & ~bits[b]),
                    "only_b_serials": corpus.examples(bits[b] & ~bits[a]),
                })
        groups.append({
            "members": group,
            "pairs": inside,
            "min_jaccard": min((p["jaccard"] for p in inside), default=1.0),
            "exact": all(p["jaccard"] == 1.0 for p in inside),
            "serials": chosen,
        })

    def group_key(g):
        first = g["members"][0]
        return (info_map.get(first, {}).get("display_name") or first).lower()

    groups.sort(key=group_key)
    for number, group in enumerate(groups, start=1):
        group["id"] = f"d{number:02d}"
    return groups


# ---- Looking for a serial that tells two patterns apart -----------------------------

def _candidates(rng: random.Random, seeds: list[str]):
    """Serials to try. First every one-digit change to each shared example — the edge
    between two patterns is usually one digit wide — then bigger nudges and random."""
    digits = "0123456789"
    seeds = [d for d in ("".join(c for c in s if c.isdigit()) for s in seeds) if len(d) == 8]
    for seed in seeds:
        for pos in range(8):
            for digit in digits:
                if digit != seed[pos]:
                    yield seed[:pos] + digit + seed[pos + 1:]
    while True:
        if seeds and rng.random() < 0.8:
            base = list(rng.choice(seeds))
            for _ in range(rng.choice([2, 2, 3, 4])):
                base[rng.randrange(8)] = rng.choice(digits)
            yield "".join(base)
        else:
            yield "".join(rng.choice(digits) for _ in range(8))


def separate(engine, a: str, b: str, seeds: list[str], seconds: float = None,
             tries: int = None, seed: int = None) -> dict:
    """Hunt for a serial one pattern matches and the other doesn't. Returns the serial and
    which pattern matched, or how many were tried without finding one. Proves nothing
    beyond what it tried."""
    rng = random.Random(seed)
    seconds = SEARCH_SECONDS if seconds is None else seconds
    tries = SEARCH_TRIES if tries is None else tries
    started = time.monotonic()
    seen, tried = set(), 0
    for digits in _candidates(rng, [s for s in seeds if s]):
        if tried >= tries or time.monotonic() - started > seconds:
            break
        if digits in seen:
            continue
        seen.add(digits)
        tried += 1
        full = f"A{digits}B"
        matched = {n for n in (a, b) if engine.execute_pattern(n, full, META).matched}
        if len(matched) == 1:
            return {"serial": full, "only": next(iter(matched)), "tried": tried,
                    "seconds": round(time.monotonic() - started, 1)}
    return {"serial": None, "only": None, "tried": tried,
            "seconds": round(time.monotonic() - started, 1)}
