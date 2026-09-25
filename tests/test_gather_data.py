"""Offline unit tests for gather_data (network downloaders are tested by use)."""
import pytest

from py3langid.train import gather_data, writer
from py3langid.train.common import DOC_CAP, MIN_DOC
from py3langid.train.gather_data import CC100_CODE, ISO3, WIKI_CODE
from py3langid.train.writer import write_docs


@pytest.fixture(autouse=True)
def cleans(monkeypatch):
    """Record clean runs of gather_data.main instead of cleaning the fake corpora."""
    calls = []
    monkeypatch.setattr(gather_data, "clean_corpus", lambda argv: calls.append("clean"))
    return calls


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


def test_tatoeba_docs_pack_per_lang(tmp_path, monkeypatch):
    """tatoeba packs sentences per lang; docs then pass the validity gate like every source"""
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

    docs = gather_data.tatoeba_docs(["en", "de"], max_docs=2)
    assert len(docs["en"]) == 2 and not docs["de"]  # 3 stubs never reach PACK_TARGET
    assert write_docs(tmp_path / "tatoeba" / "en", docs["en"], max_docs=2) == 2
    for f in (tmp_path / "tatoeba" / "en").iterdir():
        assert f.stat().st_size == DOC_CAP  # truncated, not written raw


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


def test_dedup(tmp_path):
    from py3langid.train.dedup import dedup

    line = b"x" * 80
    d1 = tmp_path / "wiki" / "aa"
    d2 = tmp_path / "cc100" / "aa"
    other = tmp_path / "wiki" / "bb"
    zxx = tmp_path / "wiki" / "zxx"
    for d in (d1, d2, other, zxx):
        d.mkdir(parents=True)
    (d1 / "doc0000.txt").write_bytes(line + b"\n\nshort\n" + b"y" * 70 + b"\npop 1234.\nsee http://b.org/x")
    (d2 / "doc0000.txt").write_bytes(line + b"\n\nshort\nunique\npop 56.\nsee https://a.org")
    (other / "doc0000.txt").write_bytes(line)  # same line, different lang: kept
    (zxx / "doc0000.txt").write_bytes(line + b"\n" + line)  # zxx untouched

    assert dedup(tmp_path) == 4
    # sorted traversal: cc100 before wiki, so d2 keeps the first occurrence
    assert (d2 / "doc0000.txt").read_bytes() == line + b"\n\nshort\nunique\npop 56.\nsee https://a.org"
    assert (d1 / "doc0000.txt").read_bytes() == b"\n" + b"y" * 70  # blank line kept, templates dropped
    assert (other / "doc0000.txt").read_bytes() == line
    assert (zxx / "doc0000.txt").read_bytes().count(line) == 2
    assert dedup(tmp_path) == 0  # idempotent


def test_valid_doc_cuts_on_codepoint_boundary():
    from py3langid.train.writer import valid_doc

    doc = valid_doc(("a" + "é" * DOC_CAP).encode())  # a naive slice ends mid-codepoint
    assert len(doc) == DOC_CAP - 1
    doc.decode("utf-8")


def test_novel_sentences_drops_repeats_and_urls():
    from py3langid.train.gather_data import novel_sentences

    seen = set()
    first = novel_sentences("HD 1 is a star. It is hot. See https://x.y/z now.", seen)
    assert first == b"HD 1 is a star.\nIt is hot.\nSee now."
    assert novel_sentences("HD 2 is a star. It is hot.", seen) == b"HD 2 is a star."
    assert novel_sentences("東京は首都。大阪も。", seen) == "東京は首都。\n大阪も。".encode()


def test_zh_routed_by_script(tmp_path):
    from py3langid.train.common import hant_majority

    trad = ("這是繁體中文的測試，我們會說這個。" * 20).encode()
    simp = ("这是简体中文的测试，我们会说这个。" * 20).encode()
    assert hant_majority(trad) and not hant_majority(simp)
    assert not hant_majority(b"no chinese here " * 40)  # tie -> primary class
    assert write_docs(tmp_path / "zh", [trad, simp], max_docs=1) == 2
    assert (tmp_path / "zh" / "doc0000.txt").read_bytes() == simp
    assert (tmp_path / "zht" / "doc0000.txt").read_bytes() == trad


def test_script_quota_admits_minority_docs_past_the_cap(tmp_path, monkeypatch):

    monkeypatch.setitem(writer.SCRIPT_QUOTA, "xx", lambda d: d.startswith(b"T"))
    docs = [b"a" * 600, b"b" * 600, b"c" * 600, b"T" * 600, b"d" * 600, b"T" * 601]
    n = write_docs(tmp_path / "xx", iter(docs), max_docs=2)
    files = sorted(p.read_bytes()[:1] for p in (tmp_path / "xx").iterdir())
    assert n == 3 and files == [b"T", b"a", b"b"]  # cap 2 + quota max_docs//2 = 1


def test_script_quota_gates_done(tmp_path, monkeypatch):

    monkeypatch.setitem(writer.SCRIPT_QUOTA, "xx", lambda d: d.startswith(b"T"))
    w = writer.DocWriter(tmp_path / "xx", max_docs=2)
    for doc in (b"a" * 600, b"b" * 600):
        w.write(doc)
    assert not w.done  # cap reached, quota not
    w.write(b"T" * 600)
    assert w.done and w.total == 3


def _fake_dump(docs):
    import bz2
    import json

    return bz2.compress("".join(json.dumps({"opening_text": d}, ensure_ascii=False) + "\n"
                                for d in docs).encode())


def _cjk_doc(sentence, i):
    return "".join(f"{sentence}第{i}篇第{k}句。" for k in range(20))  # unique sentences


def test_wiki_docs_fill_both_script_dirs(tmp_path, monkeypatch):
    """a split-script wiki fills both script dirs; a quota admits minority docs past the cap"""
    import io

    from py3langid.train import gather_data
    from py3langid.train.common import hant_majority

    trad, simp = "這是繁體中文的測試，我們會說這個", "这是简体中文的测试，我们会说这个"
    monkeypatch.setattr(gather_data, "RAW_CACHE", tmp_path / "raw")
    mixed = [_cjk_doc(s, i) for i in range(10) for s in (trad, simp)]
    monkeypatch.setattr(gather_data, "fetch", lambda *a, **k: io.BytesIO(_fake_dump(mixed)))
    docs = gather_data.wiki_docs("zh", "20260101")
    assert write_docs(tmp_path / "c/wiki/zh", docs, 3) == 6
    assert len(list((tmp_path / "c/wiki/zh").iterdir())) == 3
    assert len(list((tmp_path / "c/wiki/zht").iterdir())) == 3
    head = f"zhwiki-20260101.json.bz2.head{gather_data.WIKI_RANGE}"
    assert (tmp_path / "raw/wiki" / head).exists()

    monkeypatch.setitem(writer.SCRIPT_QUOTA, "xx", hant_majority)
    skewed = [_cjk_doc(simp, i) for i in range(8)] + [_cjk_doc(trad, i) for i in range(2)]
    monkeypatch.setattr(gather_data, "fetch", lambda *a, **k: io.BytesIO(_fake_dump(skewed)))
    docs = gather_data.wiki_docs("xx", "20260101")
    assert write_docs(tmp_path / "c/wiki/xx", docs, 4) == 6  # cap 4 + 2 Traditional


def test_single_class_writer_keeps_its_script(tmp_path):
    """per-class sources (topup, cc100 zh-Hant) skip docs of the other script"""
    trad = ("這是繁體中文的測試，我們會說這個。" * 20).encode()
    simp = ("这是简体中文的测试，我们会说这个。" * 20).encode()
    assert write_docs(tmp_path / "zht", [trad, simp, trad], max_docs=5, split=False) == 2
    assert not (tmp_path / "zh").exists()
    assert write_docs(tmp_path / "zh", [trad, simp], max_docs=5, split=False) == 1


def test_tatoeba_docs_keep_final_partial_doc(monkeypatch):
    """the last buffer below PACK_TARGET but above MIN_DOC is a doc too"""
    import io
    import tarfile

    from py3langid.train import gather_data

    rows = [("1", "deu", "x" * 99)] * 25  # 2500 bytes: one full pack + a 499-byte tail
    csv = "".join("\t".join(r) + "\n" for r in rows).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        info = tarfile.TarInfo("sentences.csv")
        info.size = len(csv)
        tar.addfile(info, io.BytesIO(csv))
    monkeypatch.setattr(gather_data, "fetch_cached", lambda *a, **k: io.BytesIO(buf.getvalue()))
    monkeypatch.setattr(gather_data, "ISO3", {"de": "deu"})
    assert [len(d) for d in gather_data.tatoeba_docs(["de"], max_docs=5)["de"]] == [1999, 499]
    assert [len(d) for d in gather_data.tatoeba_docs(["de"], max_docs=1)["de"]] == [1999]


def test_script_quota_untouched_below_the_cap(tmp_path, monkeypatch):
    """minority docs written under the cap do not consume the quota"""

    monkeypatch.setitem(writer.SCRIPT_QUOTA, "xx", lambda d: d.startswith(b"T"))
    docs = [b"T" * 600, b"T" * 601, b"a" * 600, b"b" * 600, b"T" * 602, b"c" * 600, b"T" * 603]
    assert write_docs(tmp_path / "xx", iter(docs), max_docs=4) == 6  # cap 4 + quota 2


def test_cc100_split_scripts_have_their_own_writers(tmp_path, monkeypatch):
    """zh and zht cc100 files are gathered without routing, so the writers never share a dir"""
    from py3langid.train import gather_data

    trad = ("這是繁體中文的測試，我們會說這個。" * 20).encode()
    simp = ("这是简体中文的测试，我们会说这个。" * 20).encode()
    feeds = {"zh": [trad, simp, simp, trad], "zht": [trad, trad, simp]}
    monkeypatch.setattr(gather_data, "cc100_docs", lambda lang: iter(feeds[lang]))
    gather_data.main(["--output", str(tmp_path), "--langs", "zh", "--domains", "cc100",
                      "--max-docs-per-lang", "10", "--jobs", "1"])
    assert sorted(p.read_bytes() for p in (tmp_path / "cc100/zh").iterdir()) == [simp, simp]
    assert sorted(p.read_bytes() for p in (tmp_path / "cc100/zht").iterdir()) == [trad, trad]


def test_default_langs_are_iso3(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(gather_data, "tatoeba_docs", lambda langs, max_docs: seen.append(langs) or {})
    gather_data.main(["--output", str(tmp_path), "--domains", "tatoeba"])
    assert seen == [list(ISO3)] and "zxx" not in ISO3


def test_tatoeba_split_scripts_share_one_budget(tmp_path, monkeypatch):
    """zh and zht tatoeba docs come from one cmn budget, in file order"""
    from py3langid.train import gather_data

    trad = ("這是繁體中文的測試，我們會說這個。" * 20).encode()
    simp = ("这是简体中文的测试，我们会说这个。" * 20).encode()
    feed = [simp, trad, simp, trad, simp, simp]
    monkeypatch.setattr(gather_data, "tatoeba_docs", lambda langs, n: {"zh": feed[:n], "de": [b"d" * 600] * n})
    gather_data.main(["--output", str(tmp_path), "--langs", "zh,de", "--domains", "tatoeba",
                      "--max-docs-per-lang", "3"])
    assert len(list((tmp_path / "tatoeba/zh").iterdir())) == 2
    assert len(list((tmp_path / "tatoeba/zht").iterdir())) == 1
    assert len(list((tmp_path / "tatoeba/de").iterdir())) == 3


def test_doc_rules_apply_at_write_time(tmp_path):
    """arz docs without Egyptian function words are skipped, the cell fills with passing docs"""
    msa, egy = "ذهب الرئيس إلى القاهرة. " * 30, "انا مش عايز اروح النهارده. " * 30
    docs = [d.encode() for d in (msa, egy, msa, egy + "١", egy + "٢")]
    assert write_docs(tmp_path / "arz", docs, max_docs=2) == 2
    assert all("مش" in p.read_text("utf-8") for p in (tmp_path / "arz").iterdir())


def test_manifest_records_raw_files_and_merges(tmp_path, monkeypatch):
    import hashlib
    import io
    import json
    import tarfile

    from py3langid.train import gather_data

    csv = "".join(f"{i}\tdeu\t{'x' * 99}\n" for i in range(25)).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        info = tarfile.TarInfo("sentences.csv")
        info.size = len(csv)
        tar.addfile(info, io.BytesIO(csv))
    monkeypatch.setattr(gather_data, "RAW_CACHE", tmp_path / "raw")
    monkeypatch.setattr(gather_data, "fetch", lambda *a, **k: io.BytesIO(buf.getvalue()))
    argv = ["--output", str(tmp_path / "c"), "--langs", "de", "--domains", "tatoeba"]
    gather_data.main(argv)
    manifest = tmp_path / "c" / "MANIFEST.json"
    m = json.loads(manifest.read_text("utf-8"))
    entry = m["raw"]["tatoeba/sentences.tar.bz2"]
    assert entry == {"size": len(buf.getvalue()), "sha256": hashlib.sha256(buf.getvalue()).hexdigest()}
    assert [r["langs"] for r in m["runs"]] == [["de"]] and "hf_revisions" not in m["runs"][0]

    m["raw"]["wiki/old.head"] = {"size": 1, "sha256": "0"}  # from an earlier gather
    manifest.write_text(json.dumps(m), "utf-8")
    gather_data.main([*argv[:3], "de,fr", *argv[4:]])
    m = json.loads(manifest.read_text("utf-8"))
    assert set(m["raw"]) == {"tatoeba/sentences.tar.bz2", "wiki/old.head"}
    assert [r["langs"] for r in m["runs"]] == [["de"], ["de", "fr"]]


def test_gather_cleans_last_and_before_topup(tmp_path, monkeypatch, cleans):
    import json

    from py3langid.train import topup

    monkeypatch.setattr(topup, "gather_topup", lambda *a: cleans.append("topup"))
    gather_data.main(["--output", str(tmp_path), "--langs", "de", "--domains", "topup"])
    assert cleans == ["clean", "topup", "clean"]
    assert json.loads((tmp_path / "MANIFEST.json").read_text("utf-8"))["runs"][0]["hf_revisions"] == topup.REVISION
    monkeypatch.setattr(gather_data, "cc100_docs", lambda lang: [])
    gather_data.main(["--output", str(tmp_path), "--langs", "de", "--domains", "cc100"])
    assert cleans[3:] == ["clean"]  # a re-run without topup still ends clean


def test_dedup_drops_docs_left_without_text(tmp_path):
    from py3langid.train.dedup import dedup

    d = tmp_path / "c" / "wiki" / "aa"
    d.mkdir(parents=True)
    (d / "doc0000.txt").write_bytes(b"one\ntwo")
    (d / "doc0001.txt").write_bytes(b"two\n\none")  # every line seen: dropped
    (d / "doc0002.txt").write_bytes(b"")  # empty from an earlier pass: dropped
    assert dedup(tmp_path / "c") == 2
    assert [p.name for p in d.iterdir()] == ["doc0000.txt"]
    assert (tmp_path / "c_dropped" / "wiki" / "aa" / "doc0001.txt").read_bytes() == b"two\n\none"
