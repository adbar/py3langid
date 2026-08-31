"""Smoke test for the training pipeline (end-to-end with synthetic data)."""
import pytest


@pytest.fixture
def corpus_dir(tmp_path):
    """Create a tiny corpus: 3 langs × 1 domain × 2 docs."""
    texts = {
        "en": [
            b"The quick brown fox jumps over the lazy dog and the cat sleeps",
            b"London is the capital of England and a very large city indeed",
        ],
        "de": [
            b"Der schnelle braune Fuchs springt ueber den faulen Hund heute",
            b"Berlin ist die Hauptstadt von Deutschland und eine grosse Stadt",
        ],
        "fr": [
            b"Le renard brun rapide saute par dessus le chien paresseux ici",
            b"Paris est la capitale de la France et une tres grande ville bien",
        ],
    }
    for lang, docs in texts.items():
        lang_dir = tmp_path / "corpus" / "web" / lang
        lang_dir.mkdir(parents=True)
        for i, doc in enumerate(docs):
            (lang_dir / f"doc{i}.txt").write_bytes(doc)
    return tmp_path / "corpus"


def test_training_pipeline(corpus_dir, tmp_path):
    import base64
    import bz2
    import pickle

    from py3langid.modelio import expand_nextmove, load_model
    from py3langid.train.train import main

    model_dir = tmp_path / "model"

    common_args = [
        "-j", "1",
        "--max_order", "2",
        "--min_order", "1",
        "--df_tokens", "100",
        "--feats_per_lang", "20",
        str(corpus_dir),
    ]
    main(["-m", str(model_dir)] + common_args)

    model_path = model_dir / "model.npz.xz"
    assert model_path.exists() and model_path.stat().st_size > 0

    # The shard cache was created next to the corpus
    shard_dir = corpus_dir.parent / (corpus_dir.name + ".shards")
    assert list(shard_dir.iterdir())

    # Load with the runtime
    from py3langid.langid import LanguageIdentifier

    lid = LanguageIdentifier.from_modelpath(str(model_path))
    lang, _ = lid.classify("This is a test")
    assert isinstance(lang, str)

    # Legacy loader compat: same model re-encoded as bz2+base64
    ptc, pc, classes, rows, row_index, output = load_model(model_path)[:6]
    model = (ptc, pc, classes, expand_nextmove(rows, row_index), output)
    legacy_path = tmp_path / "model_legacy"
    legacy_path.write_bytes(base64.b64encode(bz2.compress(pickle.dumps(model))))
    lid_legacy = LanguageIdentifier.from_modelpath(str(legacy_path))
    lang2, _ = lid_legacy.classify("This is a test")
    assert isinstance(lang2, str)

    # Determinism: a rerun (served from cached shards) produces the same model
    rerun_dir = tmp_path / "model_rerun"
    main(["-m", str(rerun_dir)] + common_args)
    assert (rerun_dir / "model.npz.xz").read_bytes() == model_path.read_bytes()


def test_relabel(corpus_dir, tmp_path):
    """srl dirs fold into the sr label at model assembly."""
    from py3langid.modelio import load_model
    from py3langid.train.train import main

    sr_doc = ("ово је српски текст за пробу овде").encode()
    srl_doc = b"ovo je srpski tekst za probu ovde i jos malo teksta dodato"
    for lang, doc in (("sr", sr_doc), ("srl", srl_doc)):
        d = corpus_dir / "web" / lang
        d.mkdir()
        (d / "doc0.txt").write_bytes(doc)
        (d / "doc1.txt").write_bytes(doc + b" jos")

    model_dir = tmp_path / "model"
    main(["-m", str(model_dir), "-j", "1", "--max_order", "2", "--df_tokens", "100",
          "--feats_per_lang", "20", str(corpus_dir)])
    classes = load_model(model_dir / "model.npz.xz")[2]

    assert "srl" not in classes
    assert classes.count("sr") == 2
