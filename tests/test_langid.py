
import csv
import json
import lzma
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import py3langid as langid
from py3langid.langid import (
    MODEL_DIR,
    MODEL_FILE,
    RAW_FLOOR,
    LanguageIdentifier,
    _load_identifier,
    normalize,
)
from py3langid.modelio import WordTable

NO_WORDS = WordTable(b"", np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32),
                     np.zeros(0, dtype=np.float32))


# a model load costs ~0.25s: share one per variant, undoing set_languages,
# the only mutable state, between tests
@pytest.fixture(scope='module')
def _shared_identifier():
    return LanguageIdentifier.from_modelpath(MODEL_FILE)


@pytest.fixture(scope='module')
def _shared_norm_identifier():
    return LanguageIdentifier.from_modelpath(MODEL_FILE, norm_probs=True)


@pytest.fixture
def identifier(_shared_identifier):
    yield _shared_identifier
    _shared_identifier.set_languages(None)


@pytest.fixture
def norm_identifier(_shared_norm_identifier):
    yield _shared_norm_identifier
    _shared_norm_identifier.set_languages(None)


@pytest.mark.parametrize('text,expected', [
    (b'This text is in English.', 'en'),
    ('This text is in English.', 'en'),
    ('Test Unicode sur du texte en français', 'fr'),
])
def test_classify_and_rank(text, expected):
    assert langid.classify(text)[0] == expected
    assert langid.rank(text)[0][0] == expected


def test_norm_probs(norm_identifier):
    _, prob = norm_identifier.classify('Test Unicode sur du texte en français')
    assert 0 <= prob <= 1


def test_calibration_sqrt(norm_identifier):
    """sqrt-of-bytes temperature: confidence tracks ambiguity, not saturated"""
    _, hi = norm_identifier.classify('This is clearly an English sentence with plenty of text.')
    _, lo = norm_identifier.classify('ovo je')  # short, bs/hr/sr-ambiguous
    assert lo < hi <= 1.0
    assert lo < 0.9


def test_featureless_input(identifier):
    """no features -> a flat finite floor"""
    raw = identifier.classify('#')
    # finite (stays JSON-serializable) but below any real log-probability,
    # so a caller thresholding raw scores never mistakes it for a hit
    assert raw[1] == RAW_FLOOR
    assert math.isfinite(raw[1])
    assert raw[1] < identifier.classify('This is an English sentence.')[1]


def test_unique_labels(identifier):
    """ALT_CLASS gives sr/uz/zh two columns each; the public view has one"""
    assert len(identifier._model.classes) > len(identifier.labels)
    assert len(identifier.labels) == len(set(identifier.labels))
    for dup in ('sr', 'uz', 'zh'):
        assert identifier._model.classes.count(dup) == 2
        assert identifier.labels.count(dup) == 1
    # a restricted set collapses too, however many columns it kept
    identifier.set_languages(['sr'])
    assert identifier.labels == ['sr']
    assert identifier.rank('ovo je tekst za probu') == [('sr', pytest.approx(
        identifier.classify('ovo je tekst za probu')[1]))]


def test_rank_takes_max_over_aliased_columns(identifier):
    """each label appears once in rank(), scored by its best column"""
    text = 'ovo je tekst za probu, ne znam sto to znaci'
    scores = identifier._decide(text)
    ranked = identifier.rank(text)

    assert len(ranked) == len(identifier.labels)
    assert {lang for lang, _ in ranked} == set(identifier.labels)
    enc, decoded = normalize(text)
    enc = b' ' + enc + b' '
    per_col = identifier._raw_score(enc) + identifier._word_credit(decoded)
    for lang, score in ranked:
        cols = [i for i, c in enumerate(identifier._model.classes) if c == lang]
        assert score == pytest.approx(max(float(per_col[i]) for i in cols), abs=1e-3)
    assert scores.tolist() == [s for _, s in sorted(ranked, key=lambda x: identifier.labels.index(x[0]))]
    # sr/uz really do exercise the multi-column path
    assert any(identifier._model.classes.count(c) == 2 for c in identifier.labels)


def test_rank_agrees_with_classify(identifier):
    """rank()[0] is classify()"""
    texts = ['ne znam sto to znaci', 'ovo je test', 'dobar dan', 'kaj',
             'Test Unicode sur du texte en français', 'hi', 'a']
    for text in texts:
        lang, conf = identifier.classify(text)
        assert identifier.rank(text)[0] == (lang, pytest.approx(conf)), text


def test_language_restriction(identifier):
    """a restriction narrows the class set, still classifies, and reverts"""
    full = len(identifier.labels)
    identifier.set_languages(['en', 'de'])
    assert identifier.labels == ['de', 'en']
    assert identifier.classify('This should be enough text.')[0] == 'en'
    identifier.set_languages(None)
    assert len(identifier.labels) == full


def test_unnormalized(identifier):
    _, prob = identifier.classify('Test Unicode sur du texte en français')
    assert prob < 0


def test_language_subset(identifier):
    identifier.set_languages(['de', 'en', 'fr'])
    assert identifier.classify('这样不好')[0] != 'zh'


def test_allcaps_lowering():
    '''Input is lowercased before classification'''
    assert langid.classify('CECI EST UN TEST EN FRANÇAIS')[0] == 'fr'
    assert langid.classify('DIES IST EIN DEUTSCHER TEXT')[0] == 'de'
    assert langid.classify('ЭТО РУССКИЙ ТЕКСТ ДЛЯ ТЕСТА')[0] == 'ru'
    mixed = 'NASA launched a SpaceX rocket'
    assert langid.classify(mixed) == langid.classify(mixed.lower())
    assert normalize('Ab Ç') == (b'ab \xc3\xa7', 'ab ç')


def test_bytes_str_parity():
    '''bytes and str input give identical results, all-caps included'''
    text = 'CECI EST UN TEST EN FRANÇAIS'
    assert langid.classify(text) == langid.classify(text.encode('utf8'))
    assert langid.rank(text) == langid.rank(text.encode('utf8'))


def test_truncated_bytes_reach_the_str_branch():
    '''bytes cut mid-codepoint still get lowered and NFC-normalized'''
    full = 'ЭТО РУССКИЙ ТЕКСТ ДЛЯ ТЕСТА ОПРЕДЕЛЕНИЯ ЯЗЫКА'.encode()
    cut = full[:-1]  # drops one byte of a 2-byte codepoint
    with pytest.raises(UnicodeDecodeError):
        cut.decode('utf8')
    # the partial tail is dropped, so all-caps lowering still applies
    assert langid.classify(cut)[0] == 'ru'
    assert normalize(cut)[0] == full[:-2].decode().lower().encode()
    # genuinely undecodable bytes are still passed through untouched
    assert normalize(b'\xff\xfe\xff\xfe')[0] == b'\xff\xfe\xff\xfe'


def test_empty_and_short():
    '''Feature-less input scores a finite floor, short input does not crash'''
    for empty in ('', b''):
        lang, score = langid.classify(empty)
        assert isinstance(lang, str)
        assert score == RAW_FLOOR
        # finite, so the server's JSON stays valid for strict parsers
        json.dumps({'confidence': score}, allow_nan=False)
    # digit-only input has features since the fpl700 budget: routed to zxx
    assert langid.classify('12345')[0] == 'zxx'
    lang, score = langid.classify('a')
    assert isinstance(lang, str)


def test_norm_probs_empty(norm_identifier):
    '''norm_probs=True on empty input is uniform over labels, aliased ones included'''
    probs = [p for _, p in norm_identifier.rank('')]
    assert probs == pytest.approx([1 / len(probs)] * len(probs))
    assert sum(probs) == pytest.approx(1.0)
    norm_identifier.set_languages(['en', 'sr'])
    assert [p for _, p in norm_identifier.rank('')] == pytest.approx([0.5, 0.5])
    norm_identifier.set_languages(None)
    # no n-grams but a table token: scored from its credit
    assert norm_identifier.classify('job')[0] == 'en'


def test_classify_matches_rank(norm_identifier):
    """aliased columns merge the same way in both APIs (regression)"""
    for text in ('Prema Jungovoj teoriji, m', 'Serbia II Регионална лига',
                 'Toshkent shahri markazida', 'This is an English sentence.'):
        lang, conf = norm_identifier.classify(text)
        top_lang, top_conf = norm_identifier.rank(text)[0]
        assert (lang, conf) == (top_lang, pytest.approx(top_conf, rel=1e-6))


def test_rank_sorted(identifier):
    '''rank() returns all languages sorted by descending score'''
    ranking = identifier.rank('Test Unicode sur du texte en français')
    assert ranking[0][0] == 'fr'
    scores = [s for _, s in ranking]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)
    # one entry per output label: aliased columns (sr/uz) share one
    assert len(ranking) == len(identifier.labels)
    assert len(ranking) == len({lang for lang, _ in ranking})


def test_set_languages_error(identifier):
    '''set_languages raises on unknown codes'''
    with pytest.raises(ValueError, match="Unknown language code"):
        identifier.set_languages(['xx_invalid'])
    with pytest.raises(ValueError):
        identifier.set_languages([])


def test_redirection():
    '''Test if STDIN redirection works'''
    thisdir = Path(__file__).parent
    readme_path = str(thisdir.parent / 'README.rst')
    with open(readme_path, 'rb') as f:
        readme = f.read()
    result = subprocess.check_output([sys.executable, '-m', 'py3langid.langid', '-n'],
                                     input=readme, cwd=thisdir.parent)
    assert b'en' in result and 0.5 < float(result.split()[-1].rstrip(b')')) <= 1.0


def test_cli_batch(tmp_path):
    '''Batch mode classifies files via the multiprocessing pool'''
    en = tmp_path / 'en.txt'
    en.write_bytes(b'This is an English text for testing purposes.')
    fr = tmp_path / 'fr.txt'
    fr.write_text('Ceci est un texte en français pour les tests.', encoding='utf8')
    paths = f'{en}\n{fr}\n'.encode()
    out = subprocess.check_output(['langid', '-b'], input=paths).decode()
    results = {row[0]: row[1] for row in csv.reader(out.strip().splitlines()) if row}
    assert results[str(en)] == 'en' and results[str(fr)] == 'fr'


def test_cli_external_model(tmp_path):
    '''-m loads a model from a path outside the package'''
    model_path = tmp_path / 'external.npz.xz'
    shutil.copy(MODEL_DIR / MODEL_FILE, model_path)
    result = subprocess.check_output(['langid', '-n', '-m', str(model_path)],
                                     input=b'This should be enough text.')
    assert b'en' in result and 0.5 < float(result.split()[-1].rstrip(b')')) <= 1.0
    # the path is honored, not silently replaced by the bundled model
    missing = subprocess.run(['langid', '-n', '-m', str(tmp_path / 'nope.npz.xz')],
                             input=b'This should be enough text.',
                             capture_output=True, check=False)
    assert missing.returncode != 0


def test_cli():
    '''Test console scripts entry point'''
    result = subprocess.check_output(['langid', '-n'], input=b'This should be enough text.')
    assert b'en' in result and 0.5 < float(result.split()[-1].rstrip(b')')) <= 1.0
    result = subprocess.check_output(['langid', '-n', '-l', 'bg,en,uk'], input=b'This should be enough text.')
    assert b'en' in result and 0.5 < float(result.split()[-1].rstrip(b')')) <= 1.0


@pytest.fixture
def cli(monkeypatch):
    """Run langid.main() in-process on piped stdin, returning captured stdout."""
    import io

    import py3langid.langid as mod

    def run(argv, stdin=''):
        monkeypatch.setattr(mod.sys, 'stdin', io.StringIO(stdin))
        out = io.StringIO()
        monkeypatch.setattr(mod.sys, 'stdout', out)
        mod.main(argv)
        return out.getvalue()

    yield run
    monkeypatch.setattr(mod, 'IDENTIFIER', None)  # main() sets the module global


def test_main_argv_document_and_line_mode(cli):
    """main(argv) classifies piped input as one document, or per line with --line"""
    two = 'This should be enough text.\nCeci est un texte en français.\n'
    assert cli(['-n'], two).count('\n') == 1  # one verdict for the whole document
    lines = cli(['--line'], two).strip().split('\n')
    assert [line.split("'")[1] for line in lines] == ['en', 'fr']


def test_main_argv_dist_and_langs(cli):
    """-d ranks every label, and -l restricts the ranking"""
    import ast

    out = cli(['-d', '-l', 'bg,en,uk'], 'This should be enough text.')
    ranking = ast.literal_eval(out)  # printed repr of a list of (label, score)
    langs = [lang for lang, _ in ranking]
    assert langs[0] == 'en' and sorted(langs) == ['bg', 'en', 'uk']
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True)


def test_main_argv_rejects_batch_with_serve(cli):
    """batch and serve are mutually exclusive"""
    with pytest.raises(SystemExit) as excinfo:
        cli(['-b', '-s'])
    assert excinfo.value.code == 2


def test_main_argv_unknown_lang_fails(cli):
    """-l with an unknown code surfaces the identifier's error"""
    with pytest.raises(ValueError, match='Unknown language code'):
        cli(['-l', 'en,notalang'], 'text')


def _variant(ident, **kwargs):
    """another identifier over the same arrays, no second model load"""
    words = kwargs.pop("words", NO_WORDS)  # NB only unless a table is given
    return LanguageIdentifier(ident._model._replace(words=words), **kwargs)


def test_min_confidence(identifier):
    """abstention: low calibrated confidence returns 'und'"""
    ident = _variant(identifier, norm_probs=True, min_confidence=0.5)
    lang, conf = ident.classify('This should be enough text.')
    assert lang == 'en' and conf >= 0.5
    lang, conf = ident.classify('Hi')  # too short to attribute
    assert lang == 'und' and conf < 0.5
    with pytest.raises(ValueError):
        _variant(identifier, min_confidence=0.5)


def test_from_modelpath():
    """from_modelpath loads the npz+LZMA layout from an arbitrary path"""
    ident = LanguageIdentifier.from_modelpath(MODEL_DIR / MODEL_FILE)
    assert ident.classify('This should be enough text.')[0] == 'en'
    # 0.4.0 API: package-relative path, positional norm_probs
    legacy = LanguageIdentifier.from_model_file('data/model.npz.xz', True)
    assert legacy._norm_probs
    assert legacy.classify('This should be enough text.')[0] == 'en'


def test_external_model_failure_raises(tmp_path):
    """an unusable -m path raises instead of silently falling back"""
    bad = tmp_path / 'not-a-model.npz.xz'
    bad.write_bytes(b'definitely not an xz stream')
    with pytest.raises((OSError, lzma.LZMAError, ValueError)):
        _load_identifier(str(bad))


def test_score_log1p(identifier):
    """scoring applies sublinear TF (log1p) plus class priors"""
    text = b' ' + normalize(b'This should be enough text.')[0] + b' '
    state, idxs = 0, []
    for letter in text:
        state = identifier._model.nextmove[(identifier._model.row[state] << 8) + letter]
        feat = identifier._model.output[state]  # one longest match per position
        if feat >= 0:
            idxs.append(feat)
    fc = Counter(idxs)
    idx = np.fromiter(fc.keys(), dtype=np.intp, count=len(fc))
    counts = np.fromiter(fc.values(), dtype=np.float32, count=len(fc))
    expected = np.log1p(counts) @ np.asarray(identifier._model.ptc, dtype=np.float32)[idx] \
        + identifier._model.pc
    # _raw_score is per column, before aliased columns are merged
    assert np.allclose(identifier._raw_score(text), expected,
                       rtol=1e-4)


def test_cjk_tokens_are_single_characters(identifier):
    """a CJK character in the table is credited wherever it occurs; Latin words stay whole"""
    import numpy as np

    from py3langid.langid import TOKEN_RE

    assert TOKEN_RE.findall('amazon嘅資料 ok') == ['amazon', '嘅', '資', '料', 'ok']
    col, lab = identifier._model.classes.index('yue'), identifier.labels.index('yue')
    plain = _variant(identifier)
    marked = _variant(identifier, words=WordTable("嘅".encode(), np.array([0, 1]), np.array([col]),
                                                  np.array([2.5], dtype=np.float32)))
    delta = marked._decide('佢係我嘅朋友') - plain._decide('佢係我嘅朋友')
    assert np.isclose(delta[lab], 2.5) and np.count_nonzero(delta) == 1
    assert marked._decide('plain ascii').tolist() == plain._decide('plain ascii').tolist()


def test_shipped_cjk_markers(identifier, norm_identifier):
    """Cantonese marker characters outvote the shared vocabulary, in both score modes"""
    yue = '佢係我嘅朋友，我哋一齊去睇戲'
    assert identifier.classify(yue)[0] == 'yue'
    assert norm_identifier.classify(yue)[0] == 'yue'
    assert identifier.classify('這是繁體中文的測試，我們會說這個')[0] == 'zh'


def test_word_table_credit_and_language_restriction(identifier):
    """a known word adds its credit to its columns; unknown words add nothing"""
    import numpy as np

    en, fr = identifier.labels.index('en'), identifier.labels.index('fr')
    assert identifier._model.classes.index('en') == en  # single-column labels: same index
    words = WordTable(b"bonjour\nzzqx", np.array([0, 2, 3]), np.array([fr, en, en]),
                      np.array([3.0, 0.5, 2.0], dtype=np.float32))
    plain, tabled = _variant(identifier), _variant(identifier, words=words)
    delta = tabled._decide('Bonjour bonjour') - plain._decide('Bonjour bonjour')
    assert np.isclose(delta[fr], 3.0) and np.isclose(delta[en], 0.5)
    assert np.count_nonzero(delta) == 2
    assert tabled._decide('nothing known').tolist() == plain._decide('nothing known').tolist()
    tabled.set_languages(['en', 'fr'])
    plain.set_languages(['en', 'fr'])
    delta = tabled._decide('zzqx') - plain._decide('zzqx')
    assert delta.tolist() == [2.0, 0.0]
    tabled.set_languages(None)
    assert tabled._sel is None
