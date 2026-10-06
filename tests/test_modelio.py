import io
import lzma
import pickle
import tempfile

import numpy as np
import pytest

from py3langid.modelio import Model, WordTable, load_model, save_model, sparse_counts

from .conftest import NO_WORDS


def _model(counts=((3, 0),), features=(b"ab",), classes=("en", "fr")):
    """a Model with filler priors"""
    return Model(sparse_counts(np.array(counts, dtype=np.int64).reshape(len(features), len(classes))),
                 np.full(len(classes), 0.5, dtype=np.float32), list(classes), list(features), NO_WORDS)


def test_roundtrip(tmp_path):
    # counts and gaps past the uint8 codes take the escape path
    counts = np.zeros((300, 3), dtype=np.int64)
    counts[0, 0], counts[299, 1], counts[5, 2], counts[6, 2] = 7, 70000, 255, 254
    features = [b"f%03d" % i for i in range(300)]
    pc = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    classes = ["en", "fr", "zh"]

    path = tmp_path / "model.npz.xz"
    save_model(path, Model(sparse_counts(counts), pc, classes, features, NO_WORDS))
    counts2, pc2, classes2, features2, words2 = load_model(path)
    assert words2.vocab == b"" and words2.vals.size == 0
    dense = np.zeros(counts.T.size, dtype=np.int64)
    dense[counts2.index] = counts2.values
    assert np.array_equal(dense.reshape(3, 300).T, counts) and np.array_equal(pc2, pc)
    assert classes2 == classes and features2 == features

    # the same values in other dtypes produce the same file
    save_model(path.with_suffix(".b"), Model(sparse_counts(counts.astype(np.uint32)), pc.astype(np.float64),
                                             classes, features, NO_WORDS))
    assert path.with_suffix(".b").read_bytes() == path.read_bytes()


def test_no_features(tmp_path):
    """a model without n-gram features survives the roundtrip"""
    path = tmp_path / "model.npz.xz"
    save_model(path, _model(counts=np.zeros((0, 2)), features=()))
    counts, _, classes, features, _ = load_model(path)
    assert counts.index.size == 0 and features == [] and classes == ["en", "fr"]


def test_counts_shape_checked(tmp_path):
    with pytest.raises(ValueError, match="counts beyond"):
        save_model(tmp_path / "m.npz.xz", _model(counts=((0, 1),))._replace(features=[]))


def test_unsupported_legacy_layout_rejected(tmp_path):
    """a stored-DFA model (0.5.0 development) is refused by name, not with a bare KeyError"""
    arrays = {
        "ptc": np.zeros((4, 2), dtype=np.float16),
        "pc": np.array([0.5, 0.5], dtype=np.float32),
        "classes": np.array(["en", "fr"]),
        "nextmove": np.zeros(256, dtype=np.uint16),
        "nextmove_row": np.zeros(1, dtype=np.uint16),
        "out_feat": np.full(1, -1, dtype=np.int32),
        "wt_vocab": np.zeros(0, dtype=np.uint8),
    }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    path = tmp_path / "legacy.npz.xz"
    path.write_bytes(lzma.compress(buf.getvalue(), preset=1))
    with pytest.raises(ValueError, match="feat_bytes.*nb_gap"):
        load_model(path)


def test_load_rejects_pickled_model(tmp_path):
    """a 0.3.0-style pickled model gets the layout error, not numpy's pickle error"""
    path = tmp_path / "model.plzma"
    path.write_bytes(lzma.compress(pickle.dumps(([0.5], ["en"]))))
    with pytest.raises(ValueError, match="unsupported model layout, not an npz"):
        load_model(path)


def test_load_leaves_no_temp_file(tmp_path, monkeypatch):
    """the loader cleans up its scratch file (dropping the name of one it still
    holds open is a PermissionError on Windows)"""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    path = tmp_path / "m.npz.xz"
    save_model(path, _model())
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    load_model(path)
    assert list(scratch.iterdir()) == []


def test_words_roundtrip(tmp_path):
    """the word table survives the roundtrip with snapped credits, zero credits dropped"""
    path = tmp_path / "m.npz.xz"
    words = WordTable(b"bonjour\nhello\nsalut", np.array([0, 1, 3, 4]), np.array([1, 0, 1, 0]),
                      np.array([2.55, 5.1, 0.01, 0.01], dtype=np.float32))
    save_model(path, _model()._replace(words=words))
    *_, loaded = load_model(path)
    assert loaded[0] == b"bonjour\nhello\nsalut" and loaded[1].tolist() == [0, 1, 2, 2]
    assert loaded[2].tolist() == [1, 0]
    assert np.allclose(loaded[3], [2.55, 5.1], atol=0.02)


def test_credits_snapped_to_levels(tmp_path):
    """saved credits take at most CREDIT_LEVELS + 1 values, within a few percent"""
    from py3langid.modelio import CREDIT_LEVELS
    path = tmp_path / "m.npz.xz"
    vals = np.linspace(2, 20, 500, dtype=np.float32)
    words = WordTable(b"x", np.array([0, 500]), np.zeros(500, dtype=np.int32), vals)
    save_model(path, _model()._replace(words=words))
    loaded = load_model(path).words.vals
    assert len(np.unique(loaded)) <= CREDIT_LEVELS + 1
    assert np.allclose(loaded, vals, rtol=0.05)
