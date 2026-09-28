"""Unit tests for split-script routing and the cleaning stages."""
import hashlib
from pathlib import Path

from py3langid.train import clean
from py3langid.train.common import DOC_CAP, class_of, latin_majority
from py3langid.train.dedup import dedup
from py3langid.train.rules import (
    DOC_RULES,
    apply_rules,
    not_devanagari,
    unpointed,
    zulu_test,
)
from py3langid.train.zxx import DOCS_PER_DOMAIN, DOMAIN_SEEDS, ensure_zxx

from .conftest import write_corpus


def test_latin_majority():
    assert latin_majority("Republika Srbija je država".encode())
    assert not latin_majority("Република Србија је држава".encode())
    assert not latin_majority(b"12345 ...")  # no letters -> not Latin-majority


def test_class_of():
    assert class_of("sr", "Република".encode()) == "sr"
    assert class_of("sr", b"Republika") == "srl"
    assert class_of("hr", b"Republika") == "hr"


XH = "Abantu abaninzi bathetha isiXhosa eMpuma Koloni, kwaye ulwimi lusetyenziswa ezikolweni nakwimithombo yeendaba. " * 2
ZU = "Abantu abaningi bakhuluma isiZulu KwaZulu-Natali, futhi ulimi lusetshenziswa ezikoleni nasemithonjeni yezindaba. " * 2
GOM = "गोंयांत कोंकणी भास उलयतात आनी ती राज्याची अधिकृत भास जावन आसा. गोंयच्या लोकांक आपली भास खूब मोगाची. " * 2
MR = "Mumbai ही महाराष्ट्राची राजधानी आहे आणि ते भारतातील सर्वात मोठे शहर आहे. येथे अनेक लोक राहतात आणि काम करतात."
KOK_LATN = "Goyant konknni bhas uloitat ani ti rajyachi odhikrut bhas zaun asa. Goyche lok apli bhas khub mogachi mhonntat. " * 2


def test_not_devanagari():
    assert not not_devanagari(GOM.encode())
    assert not not_devanagari(MR.encode())  # a Latin name in Devanagari text stays
    assert not_devanagari(KOK_LATN.encode())
    assert not not_devanagari(b"1234 ...")  # no letters


def test_zulu_test(tmp_path):
    """Xhosa counted outside cc100, Zulu everywhere"""
    write_corpus(tmp_path, [("wiki", "xh", XH), ("wiki", "zu", ZU),
                            ("cc100", "xh", ZU)])  # cc100 Zulu must not count as Xhosa
    zulu = zulu_test(tmp_path)
    assert zulu(ZU.encode()) and not zulu(XH.encode())


def test_line_rules(tmp_path):
    """foreign lines blanked, other bytes verbatim, only cc100 xh checked, short docs dropped untouched"""
    kept = XH + "\n" + ZU + "\n" + XH + "\n" + XH + "\udcff"  # stray 0xff
    write_corpus(tmp_path, [("cc100", "xh", kept), ("cc100", "xh", ZU + "\n" + XH),
                            ("wiki", "xh", XH + "\n" + ZU), ("wiki", "zu", ZU),
                            ("wiki", "gom", GOM + "\n" + KOK_LATN + "\n" + GOM + "\n" + MR)])
    dropped, stripped = apply_rules(tmp_path)
    assert dropped == {"xh": 1}
    assert stripped == {"xh": 2 * len(ZU.encode()), "gom": len(KOK_LATN.encode())}
    raw = lambda *parts: tmp_path.joinpath(*parts).read_bytes()
    assert raw("cc100", "xh", "doc0000.txt") == kept.replace(ZU, "").encode("utf-8", "surrogateescape")
    assert raw("wiki", "xh", "doc0000.txt") == (XH + "\n" + ZU).encode()
    assert KOK_LATN.encode() not in raw("wiki", "gom", "doc0000.txt")
    dead = tmp_path.parent / (tmp_path.name + "_dropped") / "cc100" / "xh" / "doc0001.txt"
    assert dead.read_bytes() == (ZU + "\n" + XH).encode()


def test_clean(tmp_path):
    """clean runs every stage: zxx written, Zulu stripped from cc100 xh only, duplicate line removed"""
    dup = "Dieser Satz steht in zwei Dokumenten und wird beim zweiten Mal entfernt."
    corpus = tmp_path / "corpus"
    write_corpus(corpus, [("cc100", "xh", XH * 3 + "\n" + ZU), ("wiki", "xh", XH), ("wiki", "zu", ZU * 3),
                          ("cc100", "de", ZU + "\n" + dup + "\n" + dup + "x" * 400),
                          ("wiki", "de", dup + "\n" + "Der Fuchs springt über den Hund. " * 20),
                          ("wiki", "gom", MR + "\n" + GOM)])
    clean.main([str(corpus)])
    assert len(list((corpus / "wiki" / "zxx").glob("*.txt"))) == 300
    read = lambda d, lang: (corpus / d / lang / "doc0000.txt").read_text("utf-8")
    assert ZU not in read("cc100", "xh") and XH in read("cc100", "xh")
    assert ZU in read("cc100", "de")  # not a ruled class
    assert MR in read("wiki", "gom")  # Devanagari kept
    assert sum(dup in read(d, "de") for d in ("cc100", "wiki")) == 1


def test_doc_rules():
    pointed = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ"
    modern = "בראשית ברא אלוהים את השמים ואת הארץ"
    assert not unpointed(pointed)
    assert unpointed(modern)
    assert unpointed("no Hebrew letters")
    assert not DOC_RULES["arz"]("انا مش عايز اروح النهارده")
    assert DOC_RULES["arz"]("ذهب الرئيس إلى القاهرة لحضور المؤتمر")
    assert not DOC_RULES["yue"]("佢係我嘅朋友")
    assert DOC_RULES["yue"]("他是我的朋友")


def test_zulu_test_keeps_lines_without_both_sides(tmp_path):
    line = "Abantu bonke bazalwa bekhululekile begxilile ngesidima nangamalungelo. " * 3
    write_corpus(tmp_path, [("cc100", "xh", "\n".join([line] * 8))])
    assert apply_rules(tmp_path) == ({}, {})  # no zu, no other xh
    write_corpus(tmp_path, [("wiki", "zu", line)])
    assert apply_rules(tmp_path) == ({}, {})  # no xh outside cc100


def test_dedup(tmp_path):
    line = b"x" * 80
    d1, d2, other, zxx = (Path(p) for *_, p in write_corpus(tmp_path, [
        ("wiki", "aa", line + b"\n\nshort\n" + b"y" * 70 + b"\npop 1234.\nsee http://b.org/x"),
        ("cc100", "aa", line + b"\n\nshort\nunique\npop 56.\nsee https://a.org"),
        ("wiki", "bb", line),  # same line, different lang: kept
        ("wiki", "zxx", line + b"\n" + line),  # zxx untouched
    ]))

    assert dedup(tmp_path) == 4
    # sorted traversal: cc100 before wiki, so d2 keeps the first occurrence
    assert d2.read_bytes() == line + b"\n\nshort\nunique\npop 56.\nsee https://a.org"
    assert d1.read_bytes() == b"\n" + b"y" * 70  # blank line kept, templates dropped
    assert other.read_bytes() == line
    assert zxx.read_bytes().count(line) == 2
    assert dedup(tmp_path) == 0  # idempotent


def test_dedup_drops_docs_left_without_text(tmp_path):
    write_corpus(tmp_path / "c", [("wiki", "aa", b"one\ntwo"),
                                  ("wiki", "aa", b"two\n\none"),  # every line seen: dropped
                                  ("wiki", "aa", b"")])  # empty from an earlier pass: dropped
    assert dedup(tmp_path / "c") == 2
    assert [p.name for p in (tmp_path / "c" / "wiki" / "aa").iterdir()] == ["doc0000.txt"]
    assert (tmp_path / "c_dropped" / "wiki" / "aa" / "doc0001.txt").read_bytes() == b"two\n\none"


def test_ensure_zxx_deterministic_and_idempotent(tmp_path):
    """fixed seeds give the same docs on every run; a filled corpus is left alone"""
    assert ensure_zxx(tmp_path) == DOCS_PER_DOMAIN * len(DOMAIN_SEEDS)
    assert ensure_zxx(tmp_path) == 0
    docs = [p.read_bytes() for p in sorted(tmp_path.glob("*/zxx/*.txt"))]
    assert all(len(d) <= DOC_CAP for d in docs)
    assert hashlib.sha256(b"".join(docs)).hexdigest() == \
        "c544dc155ca76be33bac2b5675e16e90047a57d927579e95d0c00ef96ae6fb80"
