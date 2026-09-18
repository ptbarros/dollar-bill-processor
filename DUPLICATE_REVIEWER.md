# Duplicate Reviewer — design / port plan

Status: **planned, not implemented** (2026-09-16). Ported from the DollarDetective-web
duplicate-review page. This doc is the implementation sketch, updated after reading the
web originals (now confirmed on disk in Paul's other checkout):

- `~/projects/dollardetective-web/scripts/pattern-similarity.py` (generator, 368 L)
- `~/projects/dollardetective-web/api/app/corpus.py` (comparator, 138 L)
- also useful: `api/app/drafts.py`, `api/app/draft_runner.py` (isolation), `api/app/review.py`

## Purpose

Find patterns that match the **same serials** (behavioural duplicates), compare their
overlays side by side, and record a keep/remove/unsure decision per pattern. Directly
supports the manual keep/prune review to build the desktop's new canonical core set.
Duplicates typically differ *only in the drawing*, so the overlay comparison is the
actual decision surface.

Motivating finding from the web: core "Broken Radar" (1,125 hits, "one digit off a
radar") strictly CONTAINS Nicks "Broken Radars" (279 hits, middle pair off) — 25% alike,
never group, despite near-identical names. **Same names are not the signal; behaviour is.**

## Why it's cheap on desktop

The two hardest pieces already exist:

- **`engine.execute_pattern(name, serial, metadata)` → `LuaExecutionResult` (`.matched`)**
  (`pattern_engine_v3.py:521`) — runs ONE pattern by name regardless of enable-state and
  injects its `DataFile` data automatically. This is the exact desktop equivalent of the
  web's `_lookup.lookup(...)` (a loop over `engine.execute_pattern`). See "Primitive" below.
- **`DigitPreviewWidget`** (`gui/pattern_dialog.py:2419`) — `set_serial()` /
  `set_highlights()` / `set_group_boxes()`, fed by `engine.get_digit_highlights(serial, [name])`.
  This is the web's `overlay.js` port; the comparison UI is just N of these stacked.

So: drop in the analysis engine nearly unchanged, **skip** the web SVG renderer (use the
widget), **skip** the Flask endpoints (use the engine directly).

## Primitive: run ALL patterns, not just enabled  (CORRECTION)

The corpus must contain every pattern so any two can be compared. On desktop:

- `classify_simple()` / `classify()` **filter to enabled patterns** (`pattern_engine_v3.py:379`,
  `if not info.enabled: continue`) — do NOT use them for the matrix.
- Use **`execute_pattern(name, serial, metadata)`** per pattern over `engine.lua_patterns`
  keys — ignores enable-state, injects data files. Matches the web numbers exactly.

```python
for name in engine.lua_patterns:                 # ALL patterns
    r = engine.execute_pattern(name, full_serial, PINNED_META)
    if r.matched:
        hits[name].add(serial_index)
```

(`classify_full()` at :428 also runs all patterns but by toggling the shared `enabled`
flags and restoring — avoid; `execute_pattern` is cleaner and has no global side effects.)

## Reproducibility: PIN the date metadata  (CRITICAL CORRECTION)

The web corpus passes NO metadata, so `create_context()` fills
`current_year/current_month/current_day` from `date.today()`. **16 Green Guide patterns
read those** (`cs_us_date_notes`, `cs_intl_future_date`, `cs_us_history_note`,
`cs_eu_history_note`, `cs_intl_birthday_note`, `cs_intl_leap_year_*`, +more — `grep
current_year patterns/`). So the web's committed corpus is a **snapshot of 2026-09-11**,
not reproducible: regenerate in 2027 and a "future date" note becomes a past one.

For the desktop port, pin it from the start:

```python
PINNED_META = {"current_year": 2026, "current_month": 9, "current_day": 11}
```

`create_context` uses `meta.setdefault(...)`, so pinned values cleanly override
`date.today()`. **Put this dict in the cache key** or a stale cache silently mixes two
date worlds. (The web documented the limit in commit f4b837a rather than changing it
mid-review; it'll pin on next regen, after which the two corpora are directly comparable.)

## How the corpus behaves (cache semantics)

- The corpus is "every pattern's hit/no-hit over a fixed, seeded ~20,734-serial set,
  under pinned date metadata."
- Valid **indefinitely** until a pattern is **added, removed, or its script edited**.
  Detected via a `patterns_version` = hash of pattern names + scripts (+ the pinned meta)
  → show a "regenerate" banner.
- **Enable/disable does NOT invalidate it** — the generator runs over *all* patterns via
  `execute_pattern`; enable-state is a filter on top. Normal review toggling is free.
- Comparisons after generation (Jaccard, `separate()`, overlays) read the cache in ms.

## Architecture

```
tools/pattern_similarity.py   ← adapted from web scripts/pattern-similarity.py (generator)
pattern_corpus.py             ← adapted from web api/app/corpus.py (compare / separate / groups)
gui/duplicate_dialog.py       ← NEW: master/detail UI + dual overlay
settings_manager.py           ← +decisions store (keep/remove/unsure/replaces/note)
gui/main_window.py            ← +Tools -> "Duplicate Review..." action
```

Generated data caches to the **user data dir** (next to `user_settings.yaml`), NOT the
repo.

## One file, not two  (DESIGN SIMPLIFICATION per web advice)

The web ships two JSONs:
- `pattern_hits.json` (~560 KB) — serials + `matches_per_serial` + zlib+base64 bitmap per
  pattern. **This is the durable artifact**; every compare / example-pick / `separate()`
  runs off it in ~2 ms.
- `pattern_similarity.json` (~930 KB) — the precomputed groups. This is really a *cache*
  of what the bitmaps can recompute.

Since the desktop owns the library, **generate only the bitmaps (`pattern_hits.json`) and
compute the duplicate groups ON LOAD** from them. Grouping is ~389 patterns → ~75k pairs
× a bitmap AND/popcount (µs each) = well under a second. One file to keep in step, not two.

## 1. Generator — `tools/pattern_similarity.py`

Adapt web `pattern-similarity.py`. Keep verbatim (pure, seeded):
- `corpus(rng)` — the seeded serial generation (SEED=20260911; ~4,000 random + solids /
  radars / repeaters / bookends / ladders / pairs-quads-fullhouses / limited-alphabet /
  2-3 distinct / years+dates (`today = date(2026,9,11)`) / low+high+zeros / alternators),
  then each pattern's own `Examples` appended, deduped, sorted, indexed.
- `write_hits()` — bitmap per pattern (`bits.to_bytes((total+7)//8, "little")` →
  `zlib.compress(...,9)` → base64), plus `serials` + `matches_per_serial`.

Change:
- Replace `_lookup.lookup(...)` with the `execute_pattern` loop above (ALL patterns).
- Pass `PINNED_META` (see above).
- **Drop** the `essentials` / `ess` fields entirely — Essentials was removed from the app.
- Emit `patterns_version` = hash of pattern names+scripts+pinned meta (for staleness).

`pattern_hits.json` schema (target):
```json
{
  "patterns_version": "<hash>",
  "corpus": {"serials": 20734, "seed": 20260911, "meta": {"current_year":2026,...}},
  "serials": ["A12345678B", ...],
  "matches_per_serial": [3, 0, 12, ...],
  "hits": {"RADAR": "<base64(zlib(bitmap))>", ...}
}
```

**Cost: ~13 min on 8 cores** (the pattern-running; grouping is trivial). Background job:
- **QThread + progress** (ship first) — simple/correct, but `execute_pattern` is CPU/GIL-
  bound so single-thread wall-time > 13 min.
- **multiprocessing.Pool** (optimize later) — the web splits serials into 50-chunks over
  `cpu_count()` workers, each with its own engine (`_init`), reassembled in serial order.

Regenerate is a deliberate button. Show cache timestamp. (The web keeps a dev-only
gitignored intermediate cache `api/.data/similarity-matches.json` keyed on version+seed;
optional on desktop.)

## 2. Comparator + grouping — `pattern_corpus.py`

Adapt web `corpus.py` (~verbatim) + the grouping half of `pattern-similarity.py`
(`duplicate_groups()`), since we compute groups on load:

- `Corpus`: lazy `bits(name)` = `int.from_bytes(zlib.decompress(b64decode(...)), "little")`;
  `compare(a,b)` → both/only_a/only_b counts + jaccard + telling example serials;
  `examples(bits)` = serials the fewest *other* patterns fire on (`matches_per_serial`).
- `groups()`: union-find over all pairs with jaccard ≥ 0.9 (`DUP_MIN_JACCARD`); each group
  ranked core → Nicks → The Green Guide → user (canonical first); `exact` flag when every
  pair is jaccard 1.0; up to 5 serials every member matches (prefer members' own examples).
- **`separate(a, b, seeds)`** — when two patterns match identically over the corpus, try
  every one-digit change to their shared example serials, then bigger nudges, then random
  (SEARCH_TRIES=60000 / SEARCH_SECONDS=5); name a distinguishing serial or report
  "tried N, found none." On desktop the `lookup` arg becomes a 2-pattern `execute_pattern`
  check: `{n for n in (a,b) if engine.execute_pattern(n, f"A{digits}B", PINNED_META).matched}`.
  Keep the honest wording — it proves nothing beyond what it tried.

Adaptation: read paths from user data dir; drop Flask/request coupling; `parse_serial` →
just build `f"A{digits}B"` and hand to `execute_pattern`.

## 3. Dialog — `gui/duplicate_dialog.py` (new)

Model on `CoverageDialog` (constructed with `engine`, `.exec()`'d from a `_on_*` handler).

- **Left:** group tree, members ordered core → Nicks → Green Guide → user (canonical
  first). Filter: identical / near / decided / undecided. Group list read like an index
  (alphabetical by first member's display name — web convention).
- **Right:** stack N `DigitPreviewWidget`s (one per member) on a shared serial — dropdown
  of the group's shared examples + free-type box. Seeing overlays side by side IS the decision.
- **`[Separate...]`** button → `pattern_corpus.separate()`.
- **Decision row:** keep/remove/unsure + `replaces` multiselect + note.

## 4. Decision storage + ownership (the real design decision)

Decisions must have ONE source of truth — two half-filled lists is the failure mode.

**Recommendation:** desktop is the source of truth. Store in `settings_manager.py` as
`pattern_decisions: {name: {decision, replaces, note}}` in `user_settings.yaml`
(consistent with labels/overrides; travels in Backup/Restore). Import the web's existing
decisions once via its CSV export. **Do NOT build bidirectional sync** with the web SQLite.

## 5. Menu wiring — `gui/main_window.py`

Copy the Coverage/Strap pattern (lines ~298-306):

```python
dup_action = QAction("&Duplicate Review...", self)
dup_action.setToolTip("Find patterns that match the same serials - compare their overlays side by side and decide which to keep")
dup_action.triggered.connect(self._on_duplicate_review)
tools_menu.addAction(dup_action)
```

`_on_duplicate_review`: check for cached corpus (offer to generate if missing) ->
`DuplicateDialog(engine, self).exec()`.

## Risks / watch-items

1. Generation wall-time is GIL-bound; QThread version exceeds 13 min — set UI expectations.
2. Corpus staleness after pattern edits — hash-and-detect (`patterns_version`), regenerate banner.
3. Date-metadata drift — pinned (see above); include the pinned dict in the cache key.
4. `get_digit_highlights` runs live for previews (fine for one serial × N members).
5. Patterns with `DataFile:` (low_runs, zip_codes, known_serials) — `execute_pattern`
   already injects their data, so no special case.
6. `separate()` false confidence — keep the "tried N, found none" wording.

## Phasing

| Phase | Deliverable |
|---|---|
| 1 | `pattern_corpus.py` + `tools/pattern_similarity.py` ported; CLI writes `pattern_hits.json`; compute + print groups; sanity-check group count vs web (~45). No UI. Usable for the current review immediately. |
| 2 | `gui/duplicate_dialog.py` read-only: group tree + dual-overlay viewer + Separate. |
| 3 | Decision storage + decision row + filters + CSV/JSON export. |
| 4 | QThread generator w/ progress + staleness detection + Tools-menu wiring. |

Phase 1 retires the porting risk and is directly usable before the GUI exists.
