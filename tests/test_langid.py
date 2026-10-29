import ast
import copy
import csv
import io
import json
import lzma
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import py3langid as langid
import py3langid.langid as mod
from py3langid.langid import (
    MODEL_FILE,
    RAW_FLOOR,
    TOKEN_RE,
    LanguageIdentifier,
    log_probs,
    normalize,
)
from py3langid.modelio import sparse_counts


def _variant(ident, words=None, **attrs):
    """a copy sharing the NB arrays and the DFA, with a toy word table {token: [(column, credit)]} if given"""
    v = copy.copy(ident)
    if words is not None:
        rows = list(words.values())
        v._words = (
            {tok: i for i, tok in enumerate(words)},
            np.cumsum([0] + [len(r) for r in rows]),
            np.array([c for r in rows for c, _ in r], dtype=np.intp),
            np.array([x for r in rows for _, x in r]),
        )
    for name, value in attrs.items():
        setattr(v, name, value)
    return v


# an identifier build costs ~0.3s: share one, each test gets a copy
@pytest.fixture(scope="module")
def _shared_identifier():
    return LanguageIdentifier.from_modelpath(MODEL_FILE)


@pytest.fixture
def identifier(_shared_identifier):
    return _variant(_shared_identifier)


@pytest.fixture
def norm_identifier(_shared_identifier):
    return _variant(_shared_identifier, _norm_probs=True)


@pytest.mark.parametrize(
    "text,expected",
    [
        (b"This text is in English.", "en"),
        ("This text is in English.", "en"),
        ("Test Unicode sur du texte en français", "fr"),
    ],
)
def test_classify_and_rank(text, expected):
    assert langid.classify(text)[0] == expected
    assert langid.rank(text)[0][0] == expected


def test_calibration_sqrt(norm_identifier):
    """sqrt-of-bytes temperature: confidence tracks ambiguity, not saturated"""
    _, hi = norm_identifier.classify("This is clearly an English sentence with plenty of text.")
    _, lo = norm_identifier.classify("ovo je")  # short, bs/hr/sr-ambiguous
    assert lo < hi <= 1.0
    assert lo < 0.9


def test_featureless_input_keeps_word_credit(identifier, norm_identifier):
    """'job' has no n-gram but a table credit, in both modes"""
    assert identifier._raw_score(b" job ") is None
    assert identifier.classify("job")[0] == "en"
    assert identifier.classify("job")[1] > RAW_FLOOR
    assert norm_identifier.classify("job")[0] == "en"


@pytest.mark.parametrize(
    "text,expected,without",
    [
        ("事体", "wuu", "zh"),
        ("予定", "ja", "zh"),  # pair markers: the characters say zh
        ("我們需要", "zh", None),
        ("伊可以节省", "wuu", None),
        ("全米選手権", "ja", None),
        ("米国体操連盟", "ja", None),
    ],
)
def test_han_pair_credits(identifier, monkeypatch, text, expected, without):
    """pair markers and bigram credits decide these"""
    assert identifier.classify(text)[0] == expected
    monkeypatch.setattr(mod, "HAN_PAIR_RE", re.compile("(?!)"))
    got = identifier.classify(text)[0]
    assert got == without if without else got != expected


def test_latin_in_han_text(identifier, monkeypatch):
    """Latin runs in Han-heavy text are dropped: they pull Mandarin to Wu"""
    text = "日本柔道选手、两枚奥运会金牌得主齐藤仁 (Hitoshi Saito, Japan Judo) 逝世，享年 54 岁。"
    assert identifier.classify(text)[0] == "zh"
    # undecodable bytes go too
    assert identifier.rank(b"\xff" + text.encode()) == identifier.rank(text)
    monkeypatch.setattr(mod, "strip_latin", lambda *a: None)
    assert identifier.classify(text)[0] == "wuu"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("python serial 自動重連", "  自動重連"),  # 3 letters per Han char
        ("python serials 自動重連", None),
        ("iphone and android 手机", None),  # Latin majority
        ("àéîõù àéîõa 中", None),  # the gate counts all Latin letters
        ("伊a可以节省", "伊可以节省"),  # pairs form across the cut
        ("越南总理范明政 (phạm minh chính) 将继续", "越南总理范明政 (  ) 将继续"),
        ("享年 a2001 岁", "享年  岁"),  # digits go with a Latin letter only
        ("享年 2001 岁", None),
        ("\udcffabc 中文", " 中文"),
        ("中文", None),
        ("abc", None),
    ],
)
def test_strip_latin(text, expected):
    """on normalized (lowercased) text"""
    assert mod.strip_latin(text, len(mod.HAN_RE.findall(text))) == expected


def test_latin_strip_needs_a_cjk_class(identifier):
    """no strip without a CJK class in the selection"""
    identifier.set_languages(["fr", "en", "de"])
    assert identifier.classify("Bonjour mon ami 你好吗朋友们大家")[0] == "fr"
    identifier.set_languages(["fr", "en", "ja"])
    assert identifier.rank("Bonjour mon ami 你好吗朋友们大家") == identifier.rank(
        "   你好吗朋友们大家"
    )


def test_han_pair_range():
    """pairs of BMP Han only: no Ext B, radicals, CJK symbols or kana"""
    assert mod.HAN_PAIR_RE.findall("日本語") == ["日本", "本語"]
    assert mod.HAN_PAIR_RE.findall("𠀀𠀁") == []
    assert mod.HAN_PAIR_RE.findall("〆〇⺀⺀") == []
    assert mod.HAN_PAIR_RE.findall("日の本") == []


def test_unique_labels(identifier):
    """ALT_CLASS gives sr/uz/zh two columns each; the public view has one"""
    assert len(identifier._model.classes) > len(identifier.labels)
    assert len(identifier.labels) == len(set(identifier.labels))
    for dup in ("sr", "uz", "zh"):
        assert identifier._model.classes.count(dup) == 2
        assert identifier.labels.count(dup) == 1
    # a restricted set collapses too, however many columns it kept
    identifier.set_languages(["sr"])
    assert identifier.labels == ["sr"]
    assert identifier.rank("ovo je tekst za probu") == [
        ("sr", pytest.approx(identifier.classify("ovo je tekst za probu")[1]))
    ]


def test_rank_takes_max_over_aliased_columns(identifier):
    """each label appears once in rank(), scored by its best column"""
    text = "ovo je tekst za probu, ne znam sto to znaci"
    scores = identifier._decide(text)
    ranked = identifier.rank(text)

    assert len(ranked) == len(identifier.labels)
    assert {lang for lang, _ in ranked} == set(identifier.labels)
    enc, decoded = normalize(text)
    enc = b" " + enc + b" "
    per_col = identifier._raw_score(enc) + identifier._word_credit(decoded, 0)
    for lang, score in ranked:
        cols = [i for i, c in enumerate(identifier._model.classes) if c == lang]
        assert score == pytest.approx(max(float(per_col[i]) for i in cols), abs=1e-3)
    assert scores.tolist() == [
        s for _, s in sorted(ranked, key=lambda x: identifier.labels.index(x[0]))
    ]


@pytest.mark.parametrize("norm", [False, True])
def test_rank_agrees_with_classify(identifier, norm_identifier, norm):
    """rank()[0] is classify(), aliased columns included (regression)"""
    ident = norm_identifier if norm else identifier
    for text in (
        "ne znam sto to znaci",
        "ovo je test",
        "dobar dan",
        "kaj",
        "hi",
        "a",
        "Test Unicode sur du texte en français",
        "Prema Jungovoj teoriji, m",
        "Serbia II Регионална лига",
        "Toshkent shahri markazida",
    ):
        lang, conf = ident.classify(text)
        assert ident.rank(text)[0] == (lang, pytest.approx(conf, rel=1e-6)), text


def test_language_restriction(identifier):
    """a restriction narrows the labels, rejects bad codes, and reverts"""
    full = len(identifier.labels)
    identifier.set_languages(["en", "de"])
    assert identifier.labels == ["de", "en"]
    assert identifier.classify("This should be enough text.")[0] == "en"
    identifier.set_languages(["de", "en", "fr"])
    assert identifier.classify("这样不好")[0] != "zh"
    with pytest.raises(ValueError, match="Unknown language code"):
        identifier.set_languages(["xx_invalid"])
    with pytest.raises(ValueError):
        identifier.set_languages([])
    identifier.set_languages(None)
    assert len(identifier.labels) == full


def test_module_set_languages():
    """the module-level helper restricts the shared identifier"""
    try:
        langid.set_languages(["de", "en"])
        assert [lang for lang, _ in langid.rank("This should be enough text.")] == ["en", "de"]
    finally:
        langid.set_languages(None)
    assert len(langid.rank("This should be enough text.")) > 2


def test_allcaps_lowering():
    """Input is lowercased before classification"""
    assert langid.classify("CECI EST UN TEST EN FRANÇAIS")[0] == "fr"
    assert langid.classify("DIES IST EIN DEUTSCHER TEXT")[0] == "de"
    assert langid.classify("ЭТО РУССКИЙ ТЕКСТ ДЛЯ ТЕСТА")[0] == "ru"
    mixed = "NASA launched a SpaceX rocket"
    assert langid.classify(mixed) == langid.classify(mixed.lower())
    assert normalize("Ab Ç") == (b"ab \xc3\xa7", "ab ç")


def test_bytes_str_parity():
    """bytes and str input give identical results, all-caps included"""
    text = "CECI EST UN TEST EN FRANÇAIS"
    assert langid.classify(text) == langid.classify(text.encode("utf8"))
    assert langid.rank(text) == langid.rank(text.encode("utf8"))


def test_truncated_bytes_reach_the_str_branch():
    """bytes cut mid-codepoint still get lowered and NFC-normalized"""
    full = "ЭТО РУССКИЙ ТЕКСТ ДЛЯ ТЕСТА ОПРЕДЕЛЕНИЯ ЯЗЫКА".encode()
    cut = full[:-1]  # drops one byte of a 2-byte codepoint
    with pytest.raises(UnicodeDecodeError):
        cut.decode("utf8")
    # the partial tail is dropped, so all-caps lowering still applies
    assert langid.classify(cut)[0] == "ru"
    assert normalize(cut)[0] == full[:-2].decode().lower().encode()
    # genuinely undecodable bytes are still passed through untouched
    assert normalize(b"\xff\xfe\xff\xfe")[0] == b"\xff\xfe\xff\xfe"
    # an invalid byte near the end is not a cut, the tail is kept
    assert normalize(b"CAF\xe9S")[0] == b"caf\xe9s"
    assert normalize(b"hello\xffab")[0] == b"hello\xffab"
    assert normalize(b"\xff")[0] == b"\xff"


def test_undecodable_bytes_are_lowered():
    """decodable parts of invalid UTF-8 get lowered and NFC-normalized"""
    assert langid.classify(b"DAS WETTER IST SCH\xd6N HEUTE MORGEN")[0] == "de"
    text = "ЭТО РУССКИЙ ТЕКСТ ДЛЯ ТЕСТА ОПРЕДЕЛЕНИЯ ЯЗЫКА"
    assert normalize(b"\xff" + text.encode())[0] == b"\xff" + text.lower().encode()
    assert langid.classify(b"\xff" + text.encode())[0] == "ru"
    assert normalize(b"\xff" + "CAFE\u0301".encode()) == (b"\xffcaf\xc3\xa9", "\udcffcafé")


def test_featureless_and_short(identifier):
    """Feature-less input scores a finite floor, short input does not crash"""
    for empty in ("", b"", "#"):
        lang, score = identifier.classify(empty)
        assert isinstance(lang, str)
        assert score == RAW_FLOOR
        # finite, so the server's JSON stays valid for strict parsers
        json.dumps({"confidence": score}, allow_nan=False)
    assert RAW_FLOOR < identifier.classify("This is an English sentence.")[1] < 0
    # digit-only input has features since the fpl700 budget: routed to zxx
    assert identifier.classify("12345")[0] == "zxx"
    assert isinstance(identifier.classify("a")[0], str)


def test_norm_probs_empty(norm_identifier):
    """norm_probs=True on empty input is uniform over labels, aliased ones included"""
    probs = [p for _, p in norm_identifier.rank("")]
    assert probs == pytest.approx([1 / len(probs)] * len(probs))
    assert sum(probs) == pytest.approx(1.0)
    norm_identifier.set_languages(["en", "sr"])
    assert [p for _, p in norm_identifier.rank("")] == pytest.approx([0.5, 0.5])


def test_rank_sorted(identifier):
    """rank() returns all languages sorted by descending score"""
    ranking = identifier.rank("Test Unicode sur du texte en français")
    assert ranking[0][0] == "fr"
    scores = [s for _, s in ranking]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)


def _confident_en(out):
    return b"en" in out and 0.5 < float(out.split()[-1].rstrip(b")")) <= 1.0


def test_redirection():
    """Test if STDIN redirection works"""
    thisdir = Path(__file__).parent
    readme_path = str(thisdir.parent / "README.rst")
    with open(readme_path, "rb") as f:
        readme = f.read()
    result = subprocess.check_output(
        [sys.executable, "-m", "py3langid.langid", "-n"], input=readme, cwd=thisdir.parent
    )
    assert _confident_en(result)


def test_cli_batch(tmp_path):
    """Batch mode classifies files via the multiprocessing pool"""
    en = tmp_path / "en.txt"
    en.write_bytes(b"This is an English text for testing purposes.")
    fr = tmp_path / "fr.txt"
    fr.write_text("Ceci est un texte en français pour les tests.", encoding="utf8")
    paths = f"{en}\n{fr}\n".encode()
    out = subprocess.check_output(["langid", "-b"], input=paths).decode()
    results = {row[0]: row[1] for row in csv.reader(out.strip().splitlines()) if row}
    assert results[str(en)] == "en" and results[str(fr)] == "fr"


def test_cli_batch_dist(tmp_path, identifier):
    """Batch mode with -d writes every label's score, best label second"""
    en = tmp_path / "en.txt"
    en.write_bytes(b"This is an English text for testing purposes.")
    out = subprocess.check_output(["langid", "-b", "-d"], input=f"{en}\n".encode()).decode()
    header, row = csv.reader(out.strip().splitlines())
    assert header == ["path", "language", *identifier.labels]
    scores = dict(zip(header[2:], map(float, row[2:])))
    assert row[:2] == [str(en), "en"] and max(scores, key=scores.__getitem__) == "en"


def test_cli_external_model(cli, tmp_path):
    """-m loads an outside model, an unusable one raises"""
    model_path = tmp_path / "external.npz.xz"
    shutil.copy(MODEL_FILE, model_path)
    lang, conf = ast.literal_eval(cli(["-n", "-m", str(model_path)], "This should be enough text."))
    assert lang == "en" and 0.5 < conf <= 1.0
    # the path is honored, not silently replaced by the bundled model
    bad = tmp_path / "not-a-model.npz.xz"
    bad.write_bytes(b"definitely not an xz stream")
    for path in (tmp_path / "nope.npz.xz", bad):
        with pytest.raises((OSError, lzma.LZMAError, ValueError)):
            cli(["-n", "-m", str(path)], "text")


def test_cli():
    """Test console scripts entry point"""
    assert _confident_en(
        subprocess.check_output(["langid", "-n"], input=b"This should be enough text.")
    )


@pytest.fixture
def cli(monkeypatch):
    """Run langid.main() in-process on piped stdin, returning captured stdout."""

    def run(argv, stdin=""):
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        mod.main(argv)
        return out.getvalue()

    # main() sets the module global: restored at teardown
    monkeypatch.setattr(mod, "IDENTIFIER", mod.IDENTIFIER)
    yield run


def test_main_argv_document_and_line_mode(cli):
    """main(argv) classifies piped input as one document, or per line with --line"""
    two = "This should be enough text.\nCeci est un texte en français.\n"
    assert cli(["-n"], two).count("\n") == 1  # one verdict for the whole document
    lines = cli(["--line"], two).strip().split("\n")
    assert [line.split("'")[1] for line in lines] == ["en", "fr"]


def test_main_argv_dist_and_langs(cli):
    """-d ranks every label, and -l restricts the ranking"""
    out = cli(["-d", "-l", "bg,en,uk"], "This should be enough text.")
    ranking = ast.literal_eval(out)  # printed repr of a list of (label, score)
    langs = [lang for lang, _ in ranking]
    assert langs[0] == "en" and sorted(langs) == ["bg", "en", "uk"]
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True)


def test_main_argv_rejects_batch_with_serve(cli):
    """batch and serve are mutually exclusive"""
    with pytest.raises(SystemExit) as excinfo:
        cli(["-b", "-s"])
    assert excinfo.value.code == 2


def test_main_argv_unknown_lang_fails(cli):
    """-l with an unknown code surfaces the identifier's error"""
    with pytest.raises(ValueError, match="Unknown language code"):
        cli(["-l", "en,notalang"], "text")


def test_min_confidence(norm_identifier):
    """abstention: low calibrated confidence returns 'und'"""
    ident = _variant(norm_identifier, {}, min_confidence=0.5)
    lang, conf = ident.classify("This should be enough text.")
    assert lang == "en" and conf >= 0.5
    lang, conf = ident.classify("Hi")  # too short to attribute
    assert lang == "und" and conf < 0.5
    with pytest.raises(ValueError):
        LanguageIdentifier.from_modelpath(MODEL_FILE, min_confidence=0.5)


def test_from_modelpath():
    """from_modelpath loads the npz+LZMA layout from an arbitrary path"""
    ident = LanguageIdentifier.from_modelpath(MODEL_FILE)
    assert ident.classify("This should be enough text.")[0] == "en"
    # 0.4.0 API: package-relative path, positional norm_probs
    legacy = LanguageIdentifier.from_model_file("data/model.npz.xz", True)
    assert legacy._norm_probs
    assert legacy.classify("This should be enough text.")[0] == "en"


def test_log_probs():
    """add-one smoothed log-probabilities match a dense reference"""
    dense = np.random.default_rng(0).integers(0, 3, (7, 4))
    ref = np.log1p(dense) - np.log(7 + dense.sum(0))
    assert np.allclose(log_probs(sparse_counts(dense), 7, 4), ref, rtol=1e-3, atol=0)  # float16


def test_cjk_tokens_are_single_characters(identifier):
    """a CJK character in the table is credited wherever it occurs; Latin words stay whole"""
    assert TOKEN_RE.findall("amazon嘅資料 ok") == ["amazon", "嘅", "資", "料", "ok"]
    assert TOKEN_RE.findall("परमेश्वर বাংলা בְּרֵאשִׁית") == ["परमेश्वर", "বাংলা", "בְּרֵאשִׁית"]
    # marks attach after a letter only: mark-only input has no token, a leading mark is dropped
    assert TOKEN_RE.findall("\u0301\u0301 \u093f 12\u0301") == []
    assert TOKEN_RE.findall("\u0301abc \u093fक") == ["abc", "क"]
    col, lab = identifier._model.classes.index("yue"), identifier.labels.index("yue")
    plain, marked = _variant(identifier, {}), _variant(identifier, {"嘅": [(col, 2.5)]})
    delta = marked._decide("佢係我嘅朋友") - plain._decide("佢係我嘅朋友")
    assert np.isclose(delta[lab], 2.5) and np.count_nonzero(delta) == 1
    assert marked._decide("plain ascii").tolist() == plain._decide("plain ascii").tolist()


def test_shipped_cjk_markers(identifier, norm_identifier):
    """Cantonese marker characters outvote the shared vocabulary, in both score modes"""
    yue = "佢係我嘅朋友，我哋一齊去睇戲"
    assert identifier.classify(yue)[0] == "yue"
    assert norm_identifier.classify(yue)[0] == "yue"
    assert identifier.classify("這是繁體中文的測試，我們會說這個")[0] == "zh"


def test_word_table_credit_and_language_restriction(identifier):
    """a known word adds its credit to its columns; unknown words add nothing"""
    en, fr = identifier.labels.index("en"), identifier.labels.index("fr")
    assert identifier._model.classes.index("en") == en  # single-column labels: same index
    plain = _variant(identifier, {})
    tabled = _variant(identifier, {"bonjour": [(fr, 3.0), (en, 0.5)], "zzqx": [(en, 2.0)]})
    delta = tabled._decide("Bonjour bonjour") - plain._decide("Bonjour bonjour")
    assert np.isclose(delta[fr], 3.0) and np.isclose(delta[en], 0.5)
    assert np.count_nonzero(delta) == 2
    assert tabled._decide("nothing known").tolist() == plain._decide("nothing known").tolist()
    tabled.set_languages(["en", "fr"])
    plain.set_languages(["en", "fr"])
    delta = tabled._decide("zzqx") - plain._decide("zzqx")
    assert delta.tolist() == [2.0, 0.0]


def test_word_credit_paths_agree(identifier, monkeypatch):
    """few-row Python sum and CSR bincount match, incl. a row repeating a column"""
    en, fr = identifier.labels.index("en"), identifier.labels.index("fr")
    tabled = _variant(identifier, {"bonjour": [(en, 1.5), (en, 0.25)], "zzqx": [(fr, 2.0)]})
    small = tabled._word_credit("bonjour zzqx", 0)
    assert small[en] == 1.75 and small[fr] == 2.0
    monkeypatch.setattr(mod, "SMALL_ROWS", 0)
    assert tabled._word_credit("bonjour zzqx", 0).tolist() == small.tolist()
    text = "this is a longer english sentence with many known words in it"
    vector = identifier._word_credit(text, 0)
    monkeypatch.setattr(mod, "SMALL_ROWS", 100)
    assert identifier._word_credit(text, 0).tolist() == vector.tolist()
