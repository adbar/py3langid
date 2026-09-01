"""Offline unit tests for gather_data (network downloaders are tested by use)."""
from pathlib import Path

import pytest

from py3langid.train.common import DOC_CAP, MIN_DOC
from py3langid.train.gather_data import (
    CC100_CODE,
    ISO3,
    WIKI_CODE,
    write_docs,
)


def test_write_docs(tmp_path):
    docs = [
        b"x" * (MIN_DOC - 1),      # stub, skipped
        b"a" * (DOC_CAP + 5000),   # truncated
        b"b" * 600,
        b"c" * 600,
    ]
    n = write_docs(tmp_path / "out", iter(docs), max_docs=2)
    assert n == 2
    files = sorted((tmp_path / "out").iterdir())
    assert [f.name for f in files] == ["doc0000.txt", "doc0001.txt"]
    assert files[0].stat().st_size == DOC_CAP
    assert files[1].read_bytes() == b"b" * 600


def test_write_docs_all_stubs(tmp_path):
    assert write_docs(tmp_path / "out", [b"tiny"], max_docs=5) == 0
    assert not (tmp_path / "out").exists()


def test_tatoeba_uses_the_validity_gate(tmp_path, monkeypatch):
    """tatoeba routes docs through valid_doc like every other source"""
    import io
    import tarfile

    from py3langid.train import gather_data

    # two "languages": eng packs 1 long sentence per doc, deu 1 stub per doc
    rows = [("1", "eng", "L" * (DOC_CAP + 500))] * 3 + [("2", "deu", "tiny")] * 3
    csv = "".join("\t".join(r) + "\n" for r in rows).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        info = tarfile.TarInfo("sentences.csv")
        info.size = len(csv)
        tar.addfile(info, io.BytesIO(csv))
    monkeypatch.setattr(gather_data, "fetch_cached",
                        lambda *a, **k: io.BytesIO(buf.getvalue()))
    monkeypatch.setattr(gather_data, "ISO3", {"en": "eng", "de": "deu"})

    counts = gather_data.gather_tatoeba(tmp_path, ["en", "de"], max_docs=2,
                                        per_doc=1)
    assert counts == {"en": 2}  # de produced only stubs -> nothing written
    for f in (tmp_path / "tatoeba" / "en").iterdir():
        assert f.stat().st_size == DOC_CAP  # truncated, not written raw
    assert not (tmp_path / "tatoeba" / "de").exists()


@pytest.mark.parametrize("mapping", [ISO3, CC100_CODE, WIKI_CODE])
def test_mapping_keys_are_iso_639(mapping):
    """keys are ISO 639-1, or 639-3 for langs without a 639-1 code"""
    assert all(2 <= len(k) <= 3 for k in mapping)


def test_iso3_values_are_distinct_639_3():
    assert all(len(v) == 3 for v in ISO3.values())
    assert len(set(ISO3.values())) == len(ISO3)


def test_fetch_cached(tmp_path, monkeypatch):
    import io

    from py3langid.train import gather_data

    calls = []

    def fake_fetch(url, headers=None, retries=3):
        calls.append(url)
        return io.BytesIO(b"payload")

    monkeypatch.setattr(gather_data, "fetch", fake_fetch)
    path = tmp_path / "cache" / "f.bin"
    for _ in range(2):
        with gather_data.fetch_cached("http://x", path) as resp:
            assert resp.read() == b"payload"
    assert len(calls) == 1  # second call served from disk
    assert not path.with_name(path.name + ".tmp").exists()


def test_teereader(tmp_path):
    import io

    from py3langid.train.gather_data import _TeeReader

    cache = tmp_path / "sub" / "prefix.bz2"
    tee = _TeeReader(io.BytesIO(b"abcdef"), cache)
    assert tee.read(4) == b"abcd"
    tee.finalize()
    assert cache.read_bytes() == b"abcd"
    assert not tee.tmp.exists()

    tee = _TeeReader(io.BytesIO(b"xyz"), tmp_path / "sub" / "other")
    tee.read()
    tee.discard()
    assert not tee.tmp.exists() and not tee.path.exists()


def test_dedup(tmp_path):
    from py3langid.train.dedup import dedup

    line = b"x" * 80
    d1 = tmp_path / "wiki" / "aa"
    d2 = tmp_path / "cc100" / "aa"
    other = tmp_path / "wiki" / "bb"
    zxx = tmp_path / "wiki" / "zxx"
    for d in (d1, d2, other, zxx):
        d.mkdir(parents=True)
    (d1 / "doc0000.txt").write_bytes(line + b"\nshort\n" + b"y" * 70)
    (d2 / "doc0000.txt").write_bytes(line + b"\nunique here " + b"z" * 60)
    (other / "doc0000.txt").write_bytes(line)  # same line, different lang: kept
    (zxx / "doc0000.txt").write_bytes(line + b"\n" + line)  # zxx untouched

    removed, touched, dropped = dedup(tmp_path)
    # d1 falls under MIN_DOC once the duplicate line goes, so it is dropped
    assert (removed, touched, dropped) == (1, 0, 1)
    # sorted traversal: cc100 before wiki, so d2 keeps the first occurrence
    assert (d2 / "doc0000.txt").read_bytes().startswith(line)
    assert not (d1 / "doc0000.txt").exists()
    assert (Path(str(tmp_path) + "_dropped") / "wiki" / "aa" / "doc0000.txt").exists()
    assert (other / "doc0000.txt").read_bytes() == line
    assert (zxx / "doc0000.txt").read_bytes().count(line) == 2
    assert dedup(tmp_path) == (0, 0, 0)  # idempotent
