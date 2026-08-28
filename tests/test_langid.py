
import csv
import subprocess
import sys
from pathlib import Path

import pytest

import py3langid as langid
from py3langid.langid import MODEL_FILE, LanguageIdentifier


@pytest.fixture
def identifier():
    return LanguageIdentifier.from_model_file(MODEL_FILE)


@pytest.fixture
def norm_identifier():
    return LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)


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


def test_gated_blend(identifier):
    """bundled model ships blend arrays; the blend pick yields a valid class"""
    assert identifier._blend is not None and identifier._blend_active
    blend_ptc, cluster_id = identifier._blend
    assert blend_ptc.shape[1] == len(identifier.nb_classes)
    assert set(cluster_id.tolist()) >= {-1, 0}
    text = identifier._encode('ovo je tekst za probu')
    pick = identifier._blend_pick(text, identifier._raw_score(text))
    assert 0 <= pick < len(identifier.nb_classes)


def test_blend_featureless(identifier, norm_identifier):
    """featureless input gets the same state-based blend pick in both modes"""
    raw = identifier.classify('hi')
    assert raw[0] == norm_identifier.classify('hi')[0]
    assert raw[0] != identifier.nb_classes[0]  # not the argmax-of-a-flat-score artifact
    assert raw[1] == 0.0  # flat floor, finite: stays JSON-serializable


def test_unique_labels(identifier):
    """LABEL_ALIAS gives sr/uz two columns each; the public view has one"""
    assert len(identifier.nb_classes) > len(identifier.labels)
    assert len(identifier.labels) == len(set(identifier.labels))
    for dup in ('sr', 'uz'):
        assert identifier.nb_classes.count(dup) == 2
        assert identifier.labels.count(dup) == 1
    # a restricted set collapses too, however many columns it kept
    identifier.set_languages(['sr'])
    assert identifier.labels == ['sr']
    assert identifier.rank('ovo je tekst za probu') == [('sr', pytest.approx(
        identifier.classify('ovo je tekst za probu')[1]))]


def test_rank_agrees_with_classify(identifier):
    """rank()[0] is classify(), blend override included"""
    texts = ['ne znam sto to znaci', 'ovo je test', 'dobar dan', 'kaj',
             'Test Unicode sur du texte en français', 'hi', 'a']
    for text in texts:
        lang, conf = identifier.classify(text)
        assert identifier.rank(text)[0] == (lang, pytest.approx(conf)), text


def test_blend_disabled_on_restriction(identifier):
    """language restriction turns the blend off; a reset re-enables it"""
    identifier.set_languages(['en', 'de'])
    assert not identifier._blend_active
    assert identifier.classify('This should be enough text.')[0] == 'en'
    identifier.set_languages(None)
    assert identifier._blend_active


def test_unnormalized(identifier):
    _, prob = identifier.classify('Test Unicode sur du texte en français')
    assert prob < 0


def test_language_subset(identifier):
    identifier.set_languages(['de', 'en', 'fr'])
    assert identifier.classify('这样不好')[0] != 'zh'


def test_allcaps_lowering():
    '''All-caps text should be lowered before classification'''
    assert langid.classify('CECI EST UN TEST EN FRANÇAIS')[0] == 'fr'
    assert langid.classify('DIES IST EIN DEUTSCHER TEXT')[0] == 'de'
    assert langid.classify('ЭТО РУССКИЙ ТЕКСТ ДЛЯ ТЕСТА')[0] == 'ru'
    # title case and mixed case are not lowered
    assert langid.classify('This is normal English text')[0] == 'en'
    assert langid.classify('NASA launched a SpaceX rocket')[0] == 'en'


def test_bytes_str_parity():
    '''bytes and str input give identical results, all-caps included'''
    text = 'CECI EST UN TEST EN FRANÇAIS'
    assert langid.classify(text) == langid.classify(text.encode('utf8'))
    assert langid.rank(text) == langid.rank(text.encode('utf8'))


def test_empty_and_short():
    '''Feature-less input scores a finite floor, short input does not crash'''
    import json
    for empty in ('', b'', '12345'):
        lang, score = langid.classify(empty)
        assert isinstance(lang, str)
        assert score == 0.0
        # finite, so the server's JSON stays valid for strict parsers
        json.dumps({'confidence': score}, allow_nan=False)
    lang, score = langid.classify('a')
    assert isinstance(lang, str)


def test_norm_probs_empty(norm_identifier):
    '''norm_probs=True on empty input returns uniform distribution'''
    _, prob = norm_identifier.classify('')
    assert abs(prob - 1.0 / len(norm_identifier.nb_classes)) < 1e-6


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


def test_set_languages_reset(identifier):
    '''set_languages(None) restores the full model'''
    full = len(identifier.nb_classes)
    identifier.set_languages(['en', 'fr'])
    assert len(identifier.nb_classes) == 2
    identifier.set_languages(None)
    assert len(identifier.nb_classes) == full


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
    '''-m loads a model in the modelstring format (b64 + bz2 pickle)'''
    import bz2
    import pickle
    from base64 import b64encode

    from py3langid.langid import MODEL_DIR
    from py3langid.modelio import load_model
    ptc, pc, classes, nextmove, output, _blend = load_model(MODEL_DIR / MODEL_FILE)
    raw = pickle.dumps((ptc, pc, classes, nextmove, output))
    model_path = tmp_path / 'external.model'
    model_path.write_bytes(b64encode(bz2.compress(raw, compresslevel=1)))
    result = subprocess.check_output(['langid', '-n', '-m', str(model_path)],
                                     input=b'This should be enough text.')
    assert b'en' in result and 0.5 < float(result.split()[-1].rstrip(b')')) <= 1.0


def test_cli():
    '''Test console scripts entry point'''
    result = subprocess.check_output(['langid', '-n'], input=b'This should be enough text.')
    assert b'en' in result and 0.5 < float(result.split()[-1].rstrip(b')')) <= 1.0
    result = subprocess.check_output(['langid', '-n', '-l', 'bg,en,uk'], input=b'This should be enough text.')
    assert b'en' in result and 0.5 < float(result.split()[-1].rstrip(b')')) <= 1.0


def test_min_confidence(norm_identifier):
    """abstention: low calibrated confidence returns 'und'"""
    ident = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True,
                                               min_confidence=0.5)
    lang, conf = ident.classify('This should be enough text.')
    assert lang == 'en' and conf >= 0.5
    lang, conf = ident.classify('Hi')  # too short to attribute
    assert lang == 'und' and conf < 0.5
    with pytest.raises(ValueError):
        LanguageIdentifier.from_model_file(MODEL_FILE, min_confidence=0.5)


def test_from_modelpath(tmp_path, identifier):
    """from_modelpath auto-detects npz+LZMA and pickled-LZMA layouts"""
    import lzma
    import pickle

    from py3langid.langid import MODEL_DIR
    ident = LanguageIdentifier.from_modelpath(MODEL_DIR / MODEL_FILE)
    assert ident.classify('This should be enough text.')[0] == 'en'

    # legacy pickle inside LZMA (flat ptc layout)
    from py3langid.modelio import load_model
    ptc, pc, classes, nextmove, output, _blend = load_model(MODEL_DIR / MODEL_FILE)
    plzma_path = tmp_path / 'model.plzma'
    with lzma.open(plzma_path, 'wb') as f:
        pickle.dump((ptc.ravel(), pc, classes, nextmove, output), f)
    ident2 = LanguageIdentifier.from_modelpath(plzma_path)
    assert ident2.classify('This should be enough text.')[0] == 'en'


def test_score_log1p(identifier):
    """scoring applies sublinear TF (log1p) plus class priors"""
    import numpy as np

    text = b'This should be enough text.'
    state, idxs = 0, []
    for letter in text:
        state = identifier.tk_nextmove[(state << 8) + letter]
        v = identifier.tk_output[state]
        if v:
            idxs.extend(v)
    from collections import Counter
    fc = Counter(idxs)
    idx = np.fromiter(fc.keys(), dtype=np.intp, count=len(fc))
    counts = np.fromiter(fc.values(), dtype=np.float32, count=len(fc))
    expected = np.log1p(counts) @ identifier.nb_ptc[idx] + identifier.nb_pc
    # norm_probs is off, so _decide's scores are the raw NB scores
    assert np.allclose(identifier._decide(text)[0], expected, rtol=1e-4)
