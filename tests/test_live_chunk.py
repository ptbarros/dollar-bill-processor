from pathlib import Path

from dollar_detective.live_chunk import select_live_chunk


def F(n, back=False):
    return Path(f"FastFoto_{n:05d}{'_b' if back else ''}.jpg")


def test_pairs_by_bill_not_count():
    # 014 case: back 59 never arrived.
    pending = [F(58), F(58, True), F(59), F(60), F(60, True)]
    assert select_live_chunk(pending) == [F(58), F(58, True)]   # wait on 59


def test_orphan_released_after_later_bills():
    pending = [F(59), F(60), F(60, True), F(61), F(61, True), F(62), F(62, True)]
    assert select_live_chunk(pending) == pending                 # 59 front-only


def test_batched_arrival_order():
    # FF-640 sometimes sends A,A,B,B -- still grouped per bill.
    pending = [F(1), F(2), F(1, True), F(2, True)]
    assert select_live_chunk(pending) == [F(1), F(1, True), F(2), F(2, True)]


def test_lone_tail_held_until_flush():
    pending = [F(1), F(1, True), F(2)]
    assert select_live_chunk(pending) == [F(1), F(1, True)]
    assert select_live_chunk([F(2)], flush=True) == [F(2)]


def test_sequential_naming_keeps_even_cut():
    pending = [Path("0001.jpg"), Path("0002.jpg"), Path("0003.jpg")]
    assert select_live_chunk(pending) == pending[:2]
    assert select_live_chunk(pending[:1]) == []
