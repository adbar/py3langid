import io
import lzma
import pickle
import tempfile
from array import array

import numpy as np
import pytest

from py3langid.modelio import Model, WordTable, load_model, save_model

NO_WORDS = WordTable(b"", np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32),
                     np.zeros(0, dtype=np.float32))


def _model(rows, row_index, output, classes=("en", "fr"), ptc_rows=1):
    """a Model with filler for the NB arrays"""
    return Model(np.zeros((ptc_rows, len(classes)), dtype=np.float32),
                 np.full(len(classes), 0.5, dtype=np.float32), list(classes),
                 rows, row_index, output, NO_WORDS)


def test_roundtrip(tmp_path):
    ptc = np.arange(12, dtype=np.float32).reshape(4, 3)
    pc = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    classes = ["en", "fr", "zh"]
    rows = array("H", range(512))
    row_index = array("L", [0, 1])
    output = [3, -1]  # one longest-match feature per state, -1 = none

    path = tmp_path / "model.npz.xz"
    save_model(path, Model(ptc, pc, classes, rows, row_index, output, NO_WORDS))
    ptc2, pc2, classes2, rows2, row2, output2, words2 = load_model(path)
    assert words2.vocab == b"" and words2.vals.size == 0

    assert np.array_equal(ptc2, ptc) and np.array_equal(pc2, pc)
    assert classes2 == classes
    assert rows2 == rows and list(row2) == [0, 1]
    assert output2 == [3, -1]

    # an array of the same values produces the same file
    save_model(path.with_suffix(".b"),
               Model(ptc, pc, classes, rows, row_index, array("l", output), NO_WORDS))
    assert path.with_suffix(".b").read_bytes() == path.read_bytes()


def test_empty_tk_output(tmp_path):
    '''model with no emitting states survives the roundtrip'''
    rows = array("H", range(256))
    path = tmp_path / "model.npz.xz"
    save_model(path, _model(rows, array("L", [0]), [-1], ptc_rows=0))
    _ptc2, _pc2, classes2, rows2, _row2, output2, _ = load_model(path)
    assert output2 == [-1] and classes2 == ["en", "fr"] and rows2 == rows


def test_uint32_widening(tmp_path):
    '''a DFA beyond the uint16 state ceiling round-trips via uint32'''
    rows = array("L", [1 << 16] * 256)  # state id overflows uint16
    save_model(tmp_path / "m.npz.xz", _model(rows, array("L", [0]), [0]))
    _, _, _, loaded, _, _, _ = load_model(tmp_path / "m.npz.xz")
    assert loaded.itemsize == 4
    assert list(loaded) == list(rows)


def test_rows_canonicalized(tmp_path):
    """rows are stored sorted and distinct, with the index remapped onto them"""
    # rows given in descending order, the second one used by two states
    rows = array("H", [2] * 256 + [1] * 256)
    path = tmp_path / "m.npz.xz"
    save_model(path, _model(rows, array("L", [0, 1, 1]), [0, -1, -1]))
    _, _, _, rows2, row_index, output, _ = load_model(path)
    assert list(rows2) == [1] * 256 + [2] * 256
    assert list(row_index) == [1, 0, 0]
    # a duplicate row passed in anyway is still stored once
    save_model(path, _model(array("H", [1] * 512), array("L", [0, 1]), [0, -1]))
    _, _, _, rows3, row_index3, _, _ = load_model(path)
    assert len(rows3) == 256 and list(row_index3) == [0, 0]
    assert output == [0, -1, -1]


def test_unsupported_legacy_layout_rejected(tmp_path):
    """a pre-row-dedup / pre-longest-match model is refused by name, not
    with a bare KeyError"""
    arrays = {
        "ptc": np.zeros((4, 2), dtype=np.float32),
        "pc": np.array([0.5, 0.5], dtype=np.float32),
        "classes": np.array(["en", "fr"]),
        "nextmove": np.array([1] * 256 + [2] * 256 + [0] * 256, dtype=np.uint16),
        "out_offsets": np.array([0, 2, 2, 3], dtype=np.uint32),
        "out_flat": np.array([1, 3, 0], dtype=np.uint32),
    }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    path = tmp_path / "legacy.npz.xz"
    path.write_bytes(lzma.compress(buf.getvalue(), preset=1))
    with pytest.raises(ValueError, match="nextmove_row.*out_feat"):
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
    save_model(path, _model(array("H", range(256)), array("L", [0]), [0]))
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    load_model(path)
    assert list(scratch.iterdir()) == []


def test_words_roundtrip(tmp_path):
    """the word table survives the roundtrip with 8-bit credits"""
    path = tmp_path / "m.npz.xz"
    words = WordTable(b"bonjour\nhello", np.array([0, 1, 3]), np.array([1, 0, 1]),
                      np.array([2.55, 5.1, 0.01], dtype=np.float32))
    save_model(path, _model(array("H", range(256)), array("L", [0]), [0])._replace(words=words))
    *_, loaded = load_model(path)
    assert loaded[0] == b"bonjour\nhello" and loaded[1].tolist() == [0, 1, 3]
    assert loaded[2].tolist() == [1, 0, 1]
    assert np.allclose(loaded[3], [2.55, 5.1, 0.0], atol=0.02)
