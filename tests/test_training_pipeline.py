"""Smoke test for the training pipeline (end-to-end with synthetic data)."""
from py3langid.langid import LanguageIdentifier
from py3langid.modelio import load_model, save_model
from py3langid.train.train import main

from .conftest import write_corpus

SR = "ово је српски текст за пробу овде"
TEXTS = {
    "en": ["The quick brown fox jumps over the lazy dog and the cat sleeps",
           "London is the capital of England and a very large city indeed"],
    "de": ["Der schnelle braune Fuchs springt ueber den faulen Hund heute",
           "Berlin ist die Hauptstadt von Deutschland und eine grosse Stadt"],
    "fr": ["Le renard brun rapide saute par dessus le chien paresseux ici",
           "Paris est la capitale de la France et une tres grande ville bien"],
    "sr": [SR, SR + " jos"],
    "srl": ["ovo je srpski tekst za probu ovde i jos malo teksta dodato",
            "ovo je srpski tekst za probu ovde i jos malo teksta dodato jos"],
}


def test_training_pipeline(tmp_path):
    corpus_dir = tmp_path / "corpus"
    write_corpus(corpus_dir, [("web", lang, t) for lang, texts in TEXTS.items() for t in texts])
    model_dir = tmp_path / "model"
    common_args = ["-j", "1", "--feats_per_lang", "20", str(corpus_dir)]
    main(["-m", str(model_dir)] + common_args)

    model_path = model_dir / "model.npz.xz"
    assert model_path.exists() and model_path.stat().st_size > 0

    # The shard cache was created next to the corpus
    shard_dir = corpus_dir.parent / (corpus_dir.name + ".shards")
    assert list(shard_dir.iterdir())

    # srl dirs fold into the sr label at model assembly
    classes = load_model(model_path).classes
    assert "srl" not in classes
    assert classes.count("sr") == 2

    lang, _ = LanguageIdentifier.from_modelpath(str(model_path)).classify("This is a test")
    assert isinstance(lang, str)

    # save_model and load_model are inverses: re-saving what was loaded
    # reproduces the file byte for byte
    resaved = tmp_path / "model_resaved.npz.xz"
    save_model(resaved, load_model(model_path))
    assert resaved.read_bytes() == model_path.read_bytes()
    lang2, _ = LanguageIdentifier.from_modelpath(str(resaved)).classify("This is a test")
    assert lang2 == lang

    # Determinism: a rerun (served from cached shards) produces the same model
    rerun_dir = tmp_path / "model_rerun"
    main(["-m", str(rerun_dir)] + common_args)
    assert (rerun_dir / "model.npz.xz").read_bytes() == model_path.read_bytes()
