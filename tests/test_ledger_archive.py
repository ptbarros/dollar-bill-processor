"""Archive & start fresh: the ledger moves to an archive file, the next open is empty."""
from dollar_detective.ledger import Ledger, archive_ledger


def _filled(db):
    led = Ledger(db)
    s = led.start_session(source_folder="001", app_version="test")
    for serial in ("A12345678B", "A20911890B", "A12345678B"):
        led.record(s, serial=serial, patterns=["YEAR_NOTES"], is_fancy=True)
    led.close()


def test_archive_moves_history_and_next_open_is_empty(tmp_path):
    db = tmp_path / "ledger.db"
    _filled(db)
    dest = archive_ledger(db)
    assert dest is not None and dest.parent == tmp_path / "ledger_archives"
    assert not db.exists()
    assert not (tmp_path / "ledger.db-wal").exists()
    assert not (tmp_path / "ledger.db-shm").exists()

    # The archive keeps the full history...
    old = Ledger(dest)
    st = old.stats()
    old.close()
    assert st["unique_bills"] == 2 and st["observations"] == 3

    # ...and a fresh ledger starts empty, with no "seen before".
    new = Ledger(db)
    assert new.stats()["observations"] == 0
    assert not new.seen_before("A12345678B")

    # The archive can be merged back if wanted.
    new.merge_from(dest)
    assert new.stats()["unique_bills"] == 2
    new.close()


def test_archive_without_ledger_is_noop(tmp_path):
    assert archive_ledger(tmp_path / "ledger.db") is None


def test_two_archives_same_second_get_distinct_names(tmp_path):
    db = tmp_path / "ledger.db"
    _filled(db)
    a = archive_ledger(db)
    _filled(db)
    b = archive_ledger(db)
    assert a != b and a.exists() and b.exists()
