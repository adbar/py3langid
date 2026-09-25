"""Unit tests for split-script routing and the cleaning stages."""
from py3langid.train.common import MIN_DOC, class_of, latin_majority
from py3langid.train.rules import apply_rules, not_devanagari, zulu_test
from py3langid.train.writer import write_docs


def test_latin_majority():
    assert latin_majority("Republika Srbija je država".encode())
    assert not latin_majority("Република Србија је држава".encode())
    assert not latin_majority(b"12345 ...")  # no letters -> not Latin-majority


def test_class_of():
    assert class_of("sr", "Република".encode()) == "sr"
    assert class_of("sr", b"Republika") == "srl"
    assert class_of("hr", b"Republika") == "hr"


def test_write_docs_splits_sr(tmp_path):
    cyr = "Београд је главни град Србије. ".encode() * 30
    lat = b"Beograd je glavni grad Srbije. " * 30
    assert len(cyr) >= MIN_DOC and len(lat) >= MIN_DOC
    n = write_docs(tmp_path / "sr", [cyr, lat, cyr, lat], max_docs=10)
    assert n == 4
    assert len(list((tmp_path / "sr").iterdir())) == 2
    assert len(list((tmp_path / "srl").iterdir())) == 2
    assert (tmp_path / "srl" / "doc0000.txt").read_bytes() == lat.strip()[:3000]


XH = "Abantu abaninzi bathetha isiXhosa eMpuma Koloni, kwaye ulwimi lusetyenziswa ezikolweni nakwimithombo yeendaba. " * 2
ZU = "Abantu abaningi bakhuluma isiZulu KwaZulu-Natali, futhi ulimi lusetshenziswa ezikoleni nasemithonjeni yezindaba. " * 2
GOM = "गोंयांत कोंकणी भास उलयतात आनी ती राज्याची अधिकृत भास जावन आसा. गोंयच्या लोकांक आपली भास खूब मोगाची. " * 2
MR = "Mumbai ही महाराष्ट्राची राजधानी आहे आणि ते भारतातील सर्वात मोठे शहर आहे. येथे अनेक लोक राहतात आणि काम करतात."


KOK_LATN = "Goyant konknni bhas uloitat ani ti rajyachi odhikrut bhas zaun asa. Goyche lok apli bhas khub mogachi mhonntat. " * 2


def _write(root, docs):
    for (domain, lang, name), text in docs.items():
        doc = root / domain / lang / name
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_bytes(text.encode("utf-8", "surrogateescape"))


def test_not_devanagari():
    assert not not_devanagari(GOM.encode())
    assert not not_devanagari(MR.encode())  # a Latin name in Devanagari text stays
    assert not_devanagari(KOK_LATN.encode())
    assert not not_devanagari(b"1234 ...")  # no letters


def test_zulu_test(tmp_path):
    """Xhosa counted outside cc100, Zulu everywhere"""
    _write(tmp_path, {("wiki", "xh", "doc0000.txt"): XH, ("wiki", "zu", "doc0000.txt"): ZU,
                      ("cc100", "xh", "doc0000.txt"): ZU})  # cc100 Zulu must not count as Xhosa
    zulu = zulu_test(tmp_path)
    assert zulu(ZU.encode()) and not zulu(XH.encode())


def test_line_rules(tmp_path):
    """foreign lines blanked, other bytes verbatim, only cc100 xh checked, short docs dropped untouched"""
    kept = XH + "\n" + ZU + "\n" + XH + "\n" + XH + "\udcff"  # stray 0xff
    _write(tmp_path, {("cc100", "xh", "doc0000.txt"): kept, ("cc100", "xh", "doc0001.txt"): ZU + "\n" + XH,
                      ("wiki", "xh", "doc0000.txt"): XH + "\n" + ZU, ("wiki", "zu", "doc0000.txt"): ZU,
                      ("wiki", "gom", "doc0000.txt"): GOM + "\n" + KOK_LATN + "\n" + GOM + "\n" + MR})
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
    from py3langid.train import clean

    dup = "Dieser Satz steht in zwei Dokumenten und wird beim zweiten Mal entfernt."
    corpus = tmp_path / "corpus"
    _write(corpus, {("cc100", "xh", "doc0000.txt"): XH * 3 + "\n" + ZU, ("wiki", "xh", "doc0000.txt"): XH,
                    ("wiki", "zu", "doc0000.txt"): ZU * 3,
                    ("cc100", "de", "doc0000.txt"): ZU + "\n" + dup + "\n" + dup + "x" * 400,
                    ("wiki", "de", "doc0000.txt"): dup + "\n" + "Der Fuchs springt über den Hund. " * 20,
                    ("wiki", "gom", "doc0000.txt"): MR + "\n" + GOM})
    clean.main([str(corpus)])
    assert len(list((corpus / "wiki" / "zxx").glob("*.txt"))) == 300
    read = lambda d, lang: (corpus / d / lang / "doc0000.txt").read_text("utf-8")
    assert ZU not in read("cc100", "xh") and XH in read("cc100", "xh")
    assert ZU in read("cc100", "de")  # not a ruled class
    assert MR in read("wiki", "gom")  # Devanagari kept
    assert sum(dup in read(d, "de") for d in ("cc100", "wiki")) == 1


def test_doc_rules():
    from py3langid.train.rules import not_cantonese, not_egyptian, unpointed

    pointed = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ"
    modern = "בראשית ברא אלוהים את השמים ואת הארץ"
    assert not unpointed(pointed)
    assert unpointed(modern)
    assert unpointed("no Hebrew letters")
    assert not not_egyptian("انا مش عايز اروح النهارده")
    assert not_egyptian("ذهب الرئيس إلى القاهرة لحضور المؤتمر")
    assert not not_cantonese("佢係我嘅朋友")
    assert not_cantonese("他是我的朋友")



def test_zulu_test_keeps_lines_without_both_sides(tmp_path):
    line = ("Abantu bonke bazalwa bekhululekile begxilile ngesidima nangamalungelo. " * 3).encode()
    doc = tmp_path / "cc100" / "xh" / "doc0000.txt"
    doc.parent.mkdir(parents=True)
    doc.write_bytes(b"\n".join([line] * 8))
    assert apply_rules(tmp_path) == ({}, {})  # no zu, no other xh
    zu = tmp_path / "wiki" / "zu" / "doc0000.txt"
    zu.parent.mkdir(parents=True)
    zu.write_bytes(line)
    assert apply_rules(tmp_path) == ({}, {})  # no xh outside cc100
