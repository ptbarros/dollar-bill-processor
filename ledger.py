"""
Bill Ledger — persistent per-bill history in SQLite.

A local, always-current record of every bill processed. It answers three things
from one store:
  1. "Have I seen this bill before?"  (instant, indexed lookup by serial)
  2. Lifetime stats                    (bills processed, unique, fancy, kept)
  3. The hit-rate vs keep-rate report  (same aggregation as tools/analyze_scans.py,
     but against live data instead of collected zips)

Design notes live in the module docstring rather than a separate doc so they
travel with the code:

  * A bill's identity is (serial_key, series_year, denomination). The same
    printed serial recurs across series years / districts, so serial alone is
    not unique; series_year disambiguates when plate extraction is on.
  * `observations` is an append-only log — one row per scan — so "bills
    processed" (COUNT observations) is distinct from "unique bills"
    (COUNT bills), exactly the split tools/analyze_scans.py makes.
  * "kept" == the bill was cropped at least once. That is the strong keep
    signal; `checked` (queued) is the softer one. Both are booleans on `bills`.
  * Re-classifying a bill (e.g. moving the gas-pump slider) replaces its
    pattern set WITHOUT adding an observation — it isn't a new scan.

This module is standalone (stdlib sqlite3 only) and does not import the app, so
it can be tested in isolation. Run `python ledger.py` for a self-test.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

DEFAULT_DB_NAME = "ledger.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY,
    started_at    TEXT NOT NULL,
    source_folder TEXT,
    label         TEXT,
    app_version   TEXT
);

CREATE TABLE IF NOT EXISTS bills (
    id            INTEGER PRIMARY KEY,
    serial        TEXT NOT NULL,
    serial_key    TEXT NOT NULL,
    series_year   TEXT NOT NULL DEFAULT '',
    denomination  TEXT NOT NULL DEFAULT '$1',
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    times_seen    INTEGER NOT NULL DEFAULT 1,
    is_fancy      INTEGER NOT NULL DEFAULT 0,
    checked       INTEGER NOT NULL DEFAULT 0,
    kept          INTEGER NOT NULL DEFAULT 0,
    kept_at       TEXT,
    front_plate   TEXT NOT NULL DEFAULT '',
    back_plate    TEXT NOT NULL DEFAULT '',
    potential_mule INTEGER NOT NULL DEFAULT 0,
    best_confidence REAL,
    UNIQUE (serial_key, series_year, denomination)
);

CREATE TABLE IF NOT EXISTS observations (
    id            INTEGER PRIMARY KEY,
    bill_id       INTEGER REFERENCES bills(id),
    session_id    INTEGER REFERENCES sessions(id),
    seen_at       TEXT NOT NULL,
    stack_position INTEGER,
    front_file    TEXT,
    back_file     TEXT,
    serial_read   TEXT,
    confidence    REAL,
    needs_review  INTEGER,
    error         TEXT,
    baseline_variance REAL,
    seal_x        REAL,
    seal_y        REAL,
    seal_containment REAL,
    star_detected INTEGER
);

CREATE TABLE IF NOT EXISTS bill_patterns (
    bill_id  INTEGER NOT NULL REFERENCES bills(id),
    pattern  TEXT NOT NULL,
    PRIMARY KEY (bill_id, pattern)
);

CREATE INDEX IF NOT EXISTS idx_bills_key   ON bills(serial_key);
CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(session_id);
CREATE INDEX IF NOT EXISTS idx_obs_bill    ON observations(bill_id);
CREATE INDEX IF NOT EXISTS idx_bp_pattern  ON bill_patterns(pattern);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_serial(serial: Optional[str]) -> str:
    """Canonical key for identity: upper-cased, star glyph spelled out.

    Mirrors the crop-filename sanitization ('*' -> 'star') so a star note's
    identity matches however it was recorded.
    """
    return (serial or "").strip().upper().replace("*", "STAR")


def _as_pattern_list(patterns: Union[str, Iterable[str], None]) -> list[str]:
    """Accept a list (GUI `fancy_types`) or a comma string (CSV `fancy_types`)."""
    if not patterns:
        return []
    if isinstance(patterns, str):
        parts = patterns.split(",")
    else:
        parts = list(patterns)
    out = []
    for p in parts:
        p = (p or "").strip()
        if p and p.upper() != "ALL":
            out.append(p)
    return out


class Ledger:
    """Thread-safe wrapper over the SQLite bill ledger.

    Writes are serialized with a lock so the module is safe to call from the
    processing QThread and the GUI thread. Open once, reuse; call close() on
    shutdown.
    """

    def __init__(self, db_path: Union[str, Path]):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- sessions ----------------------------------------------------------
    def start_session(self, source_folder: str = "", label: str = "",
                       app_version: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (started_at, source_folder, label, app_version)"
                " VALUES (?,?,?,?)",
                (_now(), source_folder, label, app_version),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    # -- reads -------------------------------------------------------------
    def seen_before(self, serial: str, series_year: str = "",
                    denomination: str = "$1") -> Optional[dict]:
        """Return the stored row for this bill, or None if never seen.

        Call this BEFORE record() if you want the pre-scan state for a badge;
        record() also returns the same info captured before its own upsert.
        """
        key = normalize_serial(serial)
        if not key:
            return None
        row = self._conn.execute(
            "SELECT * FROM bills WHERE serial_key=? AND series_year=? AND denomination=?",
            (key, series_year or "", denomination or "$1"),
        ).fetchone()
        return dict(row) if row else None

    # -- writes ------------------------------------------------------------
    def record(
        self,
        session_id: Optional[int],
        *,
        serial: Optional[str],
        series_year: str = "",
        denomination: str = "$1",
        patterns: Union[str, Iterable[str], None] = None,
        is_fancy: Optional[bool] = None,
        confidence: Optional[float] = None,
        needs_review: bool = False,
        error: Optional[str] = None,
        stack_position: Optional[int] = None,
        front_file: Optional[str] = None,
        back_file: Optional[str] = None,
        baseline_variance: Optional[float] = None,
        seal_x: Optional[float] = None,
        seal_y: Optional[float] = None,
        seal_containment: Optional[float] = None,
        star_detected: bool = False,
        front_plate: str = "",
        back_plate: str = "",
        potential_mule: bool = False,
    ) -> dict:
        """Log one processed bill: upsert its identity, append an observation,
        and replace its pattern set.

        Returns {'bill_id', 'seen_before', 'times_seen', 'first_seen', 'kept'}
        reflecting state BEFORE this scan — ready to drive a "seen before" badge.
        A blank/unreadable serial still logs an observation (so processed-counts
        stay honest) but creates no identity row.
        """
        key = normalize_serial(serial)
        pats = _as_pattern_list(patterns)
        if is_fancy is None:
            is_fancy = bool(pats)
        now = _now()

        with self._lock:
            c = self._conn
            bill_id = None
            prior = {"seen_before": False, "times_seen": 0,
                     "first_seen": None, "kept": False}

            if key:
                existing = c.execute(
                    "SELECT * FROM bills WHERE serial_key=? AND series_year=? "
                    "AND denomination=?",
                    (key, series_year or "", denomination or "$1"),
                ).fetchone()

                if existing:
                    bill_id = existing["id"]
                    prior = {
                        "seen_before": True,
                        "times_seen": existing["times_seen"],
                        "first_seen": existing["first_seen"],
                        "kept": bool(existing["kept"]),
                    }
                    best_conf = existing["best_confidence"]
                    if confidence is not None:
                        best_conf = max(best_conf or 0.0, confidence)
                    c.execute(
                        "UPDATE bills SET last_seen=?, times_seen=times_seen+1, "
                        "is_fancy=?, best_confidence=?, "
                        "front_plate=COALESCE(NULLIF(?,''), front_plate), "
                        "back_plate=COALESCE(NULLIF(?,''), back_plate), "
                        "potential_mule=? WHERE id=?",
                        (now, int(is_fancy), best_conf, front_plate, back_plate,
                         int(potential_mule), bill_id),
                    )
                else:
                    cur = c.execute(
                        "INSERT INTO bills (serial, serial_key, series_year, "
                        "denomination, first_seen, last_seen, times_seen, is_fancy, "
                        "front_plate, back_plate, potential_mule, best_confidence) "
                        "VALUES (?,?,?,?,?,?,1,?,?,?,?,?)",
                        (serial.strip() if serial else key, key, series_year or "",
                         denomination or "$1", now, now, int(is_fancy),
                         front_plate, back_plate, int(potential_mule), confidence),
                    )
                    bill_id = int(cur.lastrowid)

                # Replace pattern set to reflect the current classification.
                c.execute("DELETE FROM bill_patterns WHERE bill_id=?", (bill_id,))
                if pats:
                    c.executemany(
                        "INSERT OR IGNORE INTO bill_patterns (bill_id, pattern) VALUES (?,?)",
                        [(bill_id, p) for p in pats],
                    )

            c.execute(
                "INSERT INTO observations (bill_id, session_id, seen_at, "
                "stack_position, front_file, back_file, serial_read, confidence, "
                "needs_review, error, baseline_variance, seal_x, seal_y, "
                "seal_containment, star_detected) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (bill_id, session_id, now, stack_position, front_file, back_file,
                 serial, confidence, int(needs_review), error, baseline_variance,
                 seal_x, seal_y, seal_containment, int(star_detected)),
            )
            c.commit()

        prior["bill_id"] = bill_id
        return prior

    def mark_kept(self, serials: Iterable[str], series_year: str = "",
                  denomination: str = "$1") -> int:
        """Flag bills as kept (cropped). Call from results_list.mark_cropped().
        Accepts full serials (as shown); normalizes internally. Returns rows hit.
        """
        now = _now()
        n = 0
        with self._lock:
            for s in serials:
                key = normalize_serial(s)
                if not key:
                    continue
                cur = self._conn.execute(
                    "UPDATE bills SET kept=1, kept_at=COALESCE(kept_at,?) "
                    "WHERE serial_key=? AND series_year=? AND denomination=?",
                    (now, key, series_year or "", denomination or "$1"),
                )
                n += cur.rowcount
            self._conn.commit()
        return n

    def mark_checked(self, serials: Iterable[str], checked: bool = True,
                     series_year: str = "", denomination: str = "$1") -> int:
        """Flag bills as queued/checked (the softer interest signal)."""
        n = 0
        with self._lock:
            for s in serials:
                key = normalize_serial(s)
                if not key:
                    continue
                cur = self._conn.execute(
                    "UPDATE bills SET checked=? WHERE serial_key=? AND series_year=? "
                    "AND denomination=?",
                    (int(checked), key, series_year or "", denomination or "$1"),
                )
                n += cur.rowcount
            self._conn.commit()
        return n

    def reclassify(self, serial: str, patterns: Union[str, Iterable[str], None],
                   series_year: str = "", denomination: str = "$1",
                   is_fancy: Optional[bool] = None) -> bool:
        """Replace a bill's pattern set after a re-classify (no new observation)."""
        key = normalize_serial(serial)
        if not key:
            return False
        pats = _as_pattern_list(patterns)
        if is_fancy is None:
            is_fancy = bool(pats)
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM bills WHERE serial_key=? AND series_year=? "
                "AND denomination=?",
                (key, series_year or "", denomination or "$1"),
            ).fetchone()
            if not row:
                return False
            bid = row["id"]
            self._conn.execute("DELETE FROM bill_patterns WHERE bill_id=?", (bid,))
            if pats:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO bill_patterns (bill_id, pattern) VALUES (?,?)",
                    [(bid, p) for p in pats],
                )
            self._conn.execute("UPDATE bills SET is_fancy=? WHERE id=?",
                               (int(is_fancy), bid))
            self._conn.commit()
        return True

    # -- backup / merge ----------------------------------------------------
    def backup(self, dest_path: Union[str, Path]) -> Path:
        """Write a consistent snapshot of this ledger to dest_path using
        SQLite's online-backup API — safe while the DB is open, and it folds in
        the WAL so no -wal/-shm sidecars are needed alongside the copy."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(str(dest))
            try:
                self._conn.backup(target)
            finally:
                target.close()
        return dest

    def merge_from(self, src_path: Union[str, Path]) -> dict:
        """Merge another ledger DB into this one (for combining histories across
        machines). Bills upsert by identity — times_seen add, date range widens,
        is_fancy/kept/checked OR together, pattern sets union; sessions and
        observations are appended with remapped ids. Returns a small report."""
        src = sqlite3.connect(str(src_path))
        src.row_factory = sqlite3.Row
        added = updated = obs = 0
        with self._lock:
            c = self._conn
            sess_map = {}
            for s in src.execute("SELECT * FROM sessions"):
                cur = c.execute(
                    "INSERT INTO sessions (started_at, source_folder, label, app_version)"
                    " VALUES (?,?,?,?)",
                    (s["started_at"], s["source_folder"], s["label"], s["app_version"]))
                sess_map[s["id"]] = cur.lastrowid
            bill_map = {}
            for b in src.execute("SELECT * FROM bills"):
                ex = c.execute(
                    "SELECT id FROM bills WHERE serial_key=? AND series_year=? "
                    "AND denomination=?",
                    (b["serial_key"], b["series_year"], b["denomination"])).fetchone()
                if ex:
                    bid = ex["id"]; updated += 1
                    c.execute(
                        "UPDATE bills SET times_seen=times_seen+?, "
                        "first_seen=MIN(first_seen,?), last_seen=MAX(last_seen,?), "
                        "is_fancy=MAX(is_fancy,?), kept=MAX(kept,?), checked=MAX(checked,?), "
                        "kept_at=COALESCE(kept_at,?) WHERE id=?",
                        (b["times_seen"], b["first_seen"], b["last_seen"], b["is_fancy"],
                         b["kept"], b["checked"], b["kept_at"], bid))
                else:
                    cur = c.execute(
                        "INSERT INTO bills (serial, serial_key, series_year, denomination,"
                        " first_seen, last_seen, times_seen, is_fancy, checked, kept, kept_at,"
                        " front_plate, back_plate, potential_mule, best_confidence)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (b["serial"], b["serial_key"], b["series_year"], b["denomination"],
                         b["first_seen"], b["last_seen"], b["times_seen"], b["is_fancy"],
                         b["checked"], b["kept"], b["kept_at"], b["front_plate"],
                         b["back_plate"], b["potential_mule"], b["best_confidence"]))
                    bid = cur.lastrowid; added += 1
                bill_map[b["id"]] = bid
                for pr in src.execute("SELECT pattern FROM bill_patterns WHERE bill_id=?",
                                      (b["id"],)):
                    c.execute("INSERT OR IGNORE INTO bill_patterns (bill_id, pattern) VALUES (?,?)",
                              (bid, pr["pattern"]))
            for o in src.execute("SELECT * FROM observations"):
                c.execute(
                    "INSERT INTO observations (bill_id, session_id, seen_at, stack_position,"
                    " front_file, back_file, serial_read, confidence, needs_review, error,"
                    " baseline_variance, seal_x, seal_y, seal_containment, star_detected)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (bill_map.get(o["bill_id"]), sess_map.get(o["session_id"]), o["seen_at"],
                     o["stack_position"], o["front_file"], o["back_file"], o["serial_read"],
                     o["confidence"], o["needs_review"], o["error"], o["baseline_variance"],
                     o["seal_x"], o["seal_y"], o["seal_containment"], o["star_detected"]))
                obs += 1
            c.commit()
        src.close()
        return {"added_bills": added, "updated_bills": updated, "added_observations": obs}

    # -- aggregates --------------------------------------------------------
    def stats(self) -> dict:
        """Lifetime headline numbers."""
        c = self._conn
        obs = c.execute("SELECT COUNT(*) n FROM observations").fetchone()["n"]
        row = c.execute(
            "SELECT COUNT(*) uniq, COALESCE(SUM(is_fancy),0) fancy, "
            "COALESCE(SUM(kept),0) kept, COALESCE(SUM(checked),0) checked, "
            "MIN(first_seen) first, MAX(last_seen) last FROM bills"
        ).fetchone()
        sess = c.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"]
        dups = c.execute(
            "SELECT COUNT(*) n FROM bills WHERE times_seen>1"
        ).fetchone()["n"]
        return {
            "observations": obs,
            "unique_bills": row["uniq"],
            "duplicate_bills": dups,
            "fancy": row["fancy"],
            "kept": row["kept"],
            "checked": row["checked"],
            "sessions": sess,
            "first_seen": row["first"],
            "last_seen": row["last"],
        }

    def report_data(self) -> dict:
        """Aggregate for the analysis page. Same shape as the report artifact's
        DATA object, so it can feed the bundled HTML template directly:

            {total, fancy, unique_kept, patterns:[{name, aliases, hits, kept,
             hit_rate, keep_rate}, ...]}

        Patterns that fired on the identical set of bills are merged into one
        row (aliases[]) — the live duplicate-pattern / "pattern health" signal.
        """
        c = self._conn
        total = c.execute("SELECT COUNT(*) n FROM bills").fetchone()["n"]
        fancy = c.execute(
            "SELECT COALESCE(SUM(is_fancy),0) n FROM bills").fetchone()["n"]
        unique_kept = c.execute(
            "SELECT COALESCE(SUM(kept),0) n FROM bills").fetchone()["n"]

        kept_ids = {r["bill_id"] for r in c.execute(
            "SELECT DISTINCT bp.bill_id FROM bill_patterns bp "
            "JOIN bills b ON b.id=bp.bill_id WHERE b.kept=1")}
        pat_bills: dict[str, set] = {}
        for r in c.execute("SELECT pattern, bill_id FROM bill_patterns"):
            pat_bills.setdefault(r["pattern"], set()).add(r["bill_id"])

        # collapse patterns with identical bill-sets into alias groups
        groups: dict[frozenset, list[str]] = {}
        for pat, bills in pat_bills.items():
            groups.setdefault(frozenset(bills), []).append(pat)

        patterns = []
        for bills, names in groups.items():
            hits = len(bills)
            kept = len(bills & kept_ids)
            patterns.append({
                "name": sorted(names, key=len)[0],
                "aliases": sorted(names),
                "hits": hits,
                "kept": kept,
                "hit_rate": round(hits / total, 5) if total else 0.0,
                "keep_rate": round(kept / hits, 4) if hits else 0.0,
            })
        patterns.sort(key=lambda p: -p["hits"])
        return {"total": total, "fancy": fancy,
                "unique_kept": unique_kept, "patterns": patterns}


# ---------------------------------------------------------------------------
# Self-test / CLI
# ---------------------------------------------------------------------------
def _selftest() -> None:
    import tempfile, os, json
    tmp = tempfile.mkdtemp(prefix="ledger_test_")
    db = os.path.join(tmp, "ledger.db")
    led = Ledger(db)
    sess = led.start_session(source_folder="801", label="Aug strap 1",
                             app_version="test")

    # first scans
    r1 = led.record(sess, serial="A12344321B", patterns=["CS_RADAR", "REPEATER"],
                    series_year="2017", confidence=0.95, stack_position=1,
                    front_file="Dollar_001.jpg")
    assert r1["seen_before"] is False and r1["bill_id"], r1
    led.record(sess, serial="B11112222C", patterns="CS_QUAD_PAIRS",
               series_year="2013", stack_position=2, front_file="Dollar_003.jpg")
    led.record(sess, serial="C98765432D", patterns=None,  # not fancy
               stack_position=3, front_file="Dollar_005.jpg")
    led.record(sess, serial="B01430046*", patterns=["STAR"],  # star note
               stack_position=4, front_file="Dollar_007.jpg")
    led.record(sess, serial="", error="ocr_failed", stack_position=5)  # unreadable

    # re-scan the radar bill next session -> seen_before, times_seen bumps
    sess2 = led.start_session(source_folder="802", label="Aug strap 2")
    r_again = led.record(sess2, serial="a12344321b", patterns=["CS_RADAR", "REPEATER"],
                         series_year="2017", front_file="Dollar_101.jpg")
    assert r_again["seen_before"] is True and r_again["times_seen"] == 1, r_again

    # he crops two of them (keep signal) + queues one
    led.mark_kept(["A12344321B", "B01430046*"], series_year="")  # note: series mismatch below
    led.mark_kept(["A12344321B"], series_year="2017")
    led.mark_checked(["B11112222C"], series_year="2013")

    st = led.stats()
    print("STATS:", json.dumps(st, indent=2))
    assert st["observations"] == 6, st        # 5 + 1 re-scan
    assert st["unique_bills"] == 4, st         # radar, quad, plain, star
    assert st["duplicate_bills"] == 1, st      # radar re-seen
    assert st["fancy"] == 3, st

    rep = led.report_data()
    print("REPORT:", json.dumps(rep, indent=2))
    assert rep["total"] == 4, rep
    # CS_RADAR and REPEATER fired on the exact same single bill -> merged alias group
    radar = next(p for p in rep["patterns"] if "CS_RADAR" in p["aliases"])
    assert radar["aliases"] == ["CS_RADAR", "REPEATER"], radar
    assert radar["kept"] == 1, radar

    led.close()
    # reopen to confirm persistence
    led2 = Ledger(db)
    assert led2.stats()["unique_bills"] == 4
    led2.close()
    print("\nSelf-test passed. Temp DB:", db)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _selftest()
    else:
        import json
        led = Ledger(sys.argv[1])
        if "--report" in sys.argv:
            print(json.dumps(led.report_data(), indent=2))
        else:
            print(json.dumps(led.stats(), indent=2))
        led.close()
