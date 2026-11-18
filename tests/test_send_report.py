"""Send to Paul: report zip contents, scrubbing, size limit, and upload replies."""
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dollar_detective import send_report as sr

HOME = "/home/kelly"
SCRUB = sr.make_scrubber(home=HOME, user="kelly")


def _files(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "debug_log.txt").write_text(
        f"start\nopened {HOME}/Desktop/strap1/Dollar_001.jpg\nuser kelly ok\n"
        "key sk-ant-abcdefghijklmnop1234 used\n")
    (logs / "crash.log").write_text("")                       # empty -> not included
    st = tmp_path / "user_settings.yaml"
    st.write_text("ai:\n  anthropic_api_key: sk-ant-SECRETSECRETSECRET\n"
                  "  openai_api_key: abc\nui:\n  last_input_dir: /home/kelly/scans\n"
                  "  report_group_code: GREENBACK42\n")
    img = tmp_path / "Dollar_001.jpg"
    img.write_bytes(b"\xff\xd8fakejpeg")
    bill = {"serial": "B92414494G", "front_file": str(img), "back_file": str(tmp_path / "missing.jpg"),
            "series_year": "2021", "fancy_types": "4 of a Kind", "position": 8}
    return logs, st, bill


def test_report_contents_are_scrubbed(tmp_path):
    logs, st, bill = _files(tmp_path)
    out = tmp_path / "r.zip"
    m = sr.build_report(out, sender="Kelly", note="YOLO prompt?", log_paths=[logs / "debug_log.txt"],
                        crash_path=logs / "crash.log", settings_path=st, bills=[bill],
                        system={"app_version": "1.16.0", "os": "macOS"}, scrub=SCRUB)
    z = zipfile.ZipFile(out)
    names = set(z.namelist())
    assert {"manifest.json", "note.txt", "system.json", "logs/debug_log.txt",
            "settings/user_settings.yaml", "bills/01_B92414494G/front.jpg",
            "bills/01_B92414494G/result.json"} <= names
    assert "logs/crash.log" not in names                        # empty crash log skipped
    assert "bills/01_B92414494G/back.jpg" not in names          # missing file skipped
    log = z.read("logs/debug_log.txt").decode()
    assert HOME not in log and "~/Desktop" in log and "<user>" in log
    assert "sk-ant-" not in log and "<api-key removed>" in log
    sett = z.read("settings/user_settings.yaml").decode()
    assert "SECRET" not in sett and "abc" not in sett and "~/scans" in sett
    assert "GREENBACK42" not in sett                            # group code never sent
    man = json.loads(z.read("manifest.json"))
    assert man["format"] == "dd-report" and man["sender"] == "Kelly"
    assert man["app_version"] == "1.16.0"
    assert man["contents"]["bills"] == [{"serial": "B92414494G", "images": ["front.jpg"]}]
    assert m["size_bytes"] == out.stat().st_size
    row = json.loads(z.read("bills/01_B92414494G/result.json"))
    assert row["fancy_types"] == "4 of a Kind" and "front_file" not in row


def test_bills_capped_and_size_limit(tmp_path, monkeypatch):
    big = tmp_path / "big.jpg"
    big.write_bytes(b"x" * 2000)
    bills = [{"serial": f"A{i:08d}B", "front_file": str(big)} for i in range(15)]
    m = sr.build_report(tmp_path / "a.zip", sender="x", note="y", bills=bills, system={}, scrub=SCRUB)
    assert len(m["contents"]["bills"]) == sr.MAX_BILLS
    monkeypatch.setattr(sr, "MAX_REPORT_BYTES", 1000)
    with pytest.raises(sr.ReportError, match="limit"):
        sr.build_report(tmp_path / "b.zip", sender="x", note="y", bills=bills, system={}, scrub=SCRUB)


class _Handler(BaseHTTPRequestHandler):
    status, reply, seen = 201, {"id": "R-TEST01"}, {}

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Handler.seen = {"ctype": self.headers["Content-Type"], "body": body,
                         "ua": self.headers["User-Agent"],
                         "code": self.headers.get("X-DD-Group-Code")}
        self.send_response(_Handler.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(_Handler.reply).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}/api/reports"
    httpd.shutdown()


def test_upload_returns_reference_and_sends_multipart(tmp_path, server):
    z = tmp_path / "r.zip"
    sr.build_report(z, sender="s", note="n", system={}, scrub=SCRUB)
    _Handler.status, _Handler.reply = 201, {"id": "R-TEST01"}
    assert sr.upload_report(z, url=server, app_version="1.16.1",
                            group_code=" GREENBACK42 ") == "R-TEST01"
    seen = _Handler.seen
    assert seen["ctype"].startswith("multipart/form-data; boundary=")
    assert b'name="report"; filename="r.zip"' in seen["body"]
    assert z.read_bytes() in seen["body"]
    assert "DollarDetective/1.16.1" in seen["ua"]
    assert seen["code"] == "GREENBACK42"


@pytest.mark.parametrize("code,text", [(403, "group code"), (413, "too large"), (429, "wait"), (507, "storage is full"),
                                       (401, "isn't available"), (404, "isn't available"),
                                       (500, "HTTP 500")])
def test_upload_errors_are_plain_language(tmp_path, server, code, text):
    z = tmp_path / "r.zip"
    sr.build_report(z, sender="s", note="n", system={}, scrub=SCRUB)
    _Handler.status, _Handler.reply = code, {"detail": "x"}
    with pytest.raises(sr.ReportError, match=text):
        sr.upload_report(z, url=server)


def test_unreachable_server(tmp_path):
    z = tmp_path / "r.zip"
    sr.build_report(z, sender="s", note="n", system={}, scrub=SCRUB)
    with pytest.raises(sr.ReportError, match="Couldn't reach"):
        sr.upload_report(z, url="http://127.0.0.1:9/api/reports", timeout=3)


def test_screenshots_included_with_limits(tmp_path, monkeypatch):
    shots = []
    for name in ("a.png", "b.jpeg", "c.gif", "d.png", "e.png"):
        p = tmp_path / name
        p.write_bytes(b"img" * 10)
        shots.append(p)
    big = tmp_path / "big.png"
    big.write_bytes(b"x" * 50)
    monkeypatch.setattr(sr, "MAX_SCREENSHOT_BYTES", 40)
    out = tmp_path / "s.zip"
    # big is skipped (too large) and missing.png doesn't exist; numbering stays 01, 02
    m = sr.build_report(out, sender="s", note="n", system={}, scrub=SCRUB,
                        screenshots=[big, shots[0], shots[1]] + [tmp_path / "missing.png"])
    names = zipfile.ZipFile(out).namelist()
    assert m["contents"]["screenshots"] == ["01.png", "02.jpg"]
    assert "screenshots/01.png" in names and "screenshots/02.jpg" in names
    m2 = sr.build_report(tmp_path / "t.zip", sender="s", note="n", system={}, scrub=SCRUB,
                         screenshots=[shots[2], shots[0], shots[3], shots[4]])
    assert m2["contents"]["screenshots"] == ["01.png", "02.png", "03.png"]   # gif skipped, cap 3


@pytest.mark.parametrize("code,ok", [
    ("radar-love-1221", True), ("  radar-love-1221  ", True), ("copper ladder 47", True),
    ("short", False), ("x" * 41, False), ("café-ladder-47", False), ("star\tnote-12", False),
])
def test_group_code_shape(code, ok):
    assert (sr.group_code_problem(code) is None) == ok


def _review_folder(tmp_path):
    rf = tmp_path / "review"
    (rf / "sent" / "old").mkdir(parents=True)
    (rf / "sent" / "old" / "20260101_000000_Dollar_009.jpg").write_bytes(b"old")   # already sent
    for n in ("20261118_130000_Dollar_001.jpg", "20261118_130000_Dollar_002.jpg",
              "20261118_130000_serial_crop (1).png"):
        (rf / n).write_bytes(b"img")
    (rf / "review_log.csv").write_text(
        "timestamp,serial,note,confidence,patterns,front_file,files_copied\n"
        "2026-11-18T13:00:00,B39299183A,G/C wrong at /home/kelly/x,0.91,Gas Pump,Dollar_001.jpg,"
        "20261118_130000_Dollar_001.jpg; 20261118_130000_Dollar_002.jpg; 20261118_130000_serial_crop (1).png\n")
    return rf


def test_review_folder_packed_with_safe_names(tmp_path):
    rf = _review_folder(tmp_path)
    assert sr.review_count(rf) == 1 and sr.review_count(tmp_path / "nope") == 0
    out = tmp_path / "r.zip"
    m = sr.build_report(out, sender="s", note="n", system={}, scrub=SCRUB, review_folder=rf)
    z = zipfile.ZipFile(out)
    names = set(z.namelist())
    assert {"review/review_log.csv", "review/20261118_130000_Dollar_001.jpg",
            "review/20261118_130000_Dollar_002.jpg",
            "review/20261118_130000_serial_crop__1_.png"} <= names
    assert not any("Dollar_009" in n for n in names)            # sent/ archive left out
    log = z.read("review/review_log.csv").decode()
    assert "serial_crop__1_.png" in log and "crop (1)" not in log  # refs rewritten
    assert "/home/kelly" not in log                              # scrubbed
    assert m["contents"]["review"]["rows"] == 1
    assert len(m["contents"]["review"]["images"]) == 3
    man = json.loads(z.read("manifest.json"))
    assert man["contents"]["review"]["rows"] == 1


def test_no_review_folder_means_no_review_entries(tmp_path):
    out = tmp_path / "r.zip"
    m = sr.build_report(out, sender="s", note="n", system={}, scrub=SCRUB)
    assert not any(n.startswith("review/") for n in zipfile.ZipFile(out).namelist())
    assert m["contents"]["review"] == {"rows": 0, "images": []}
