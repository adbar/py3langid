from array import array

import numpy as np

from py3langid.modelio import load_model, save_model


def test_roundtrip(tmp_path):
    ptc = np.arange(12, dtype=np.float32).reshape(4, 3)
    pc = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    classes = ["en", "fr", "zh"]
    rows = array("H", range(512))
    row_index = array("L", [0, 1])
    output = [3, -1]  # one longest-match feature per state, -1 = none

    path = tmp_path / "model.npz.xz"
    save_model(path, (ptc, pc, classes, rows, row_index, output))
    ptc2, pc2, classes2, rows2, row2, output2 = load_model(path)

    assert np.array_equal(ptc2, ptc) and np.array_equal(pc2, pc)
    assert classes2 == classes
    assert rows2 == rows and list(row2) == [0, 1]
    assert output2 == [3, -1]

    # an array of the same values produces the same file
    save_model(path.with_suffix(".b"),
               (ptc, pc, classes, rows, row_index, array("l", output)))
    assert path.with_suffix(".b").read_bytes() == path.read_bytes()


def test_empty_tk_output(tmp_path):
    '''model with no emitting states survives the roundtrip'''
    ptc = np.zeros((0, 2), dtype=np.float32)
    pc = np.array([0.5, 0.5], dtype=np.float32)
    rows = array("H", range(256))
    path = tmp_path / "model.npz.xz"
    save_model(path, (ptc, pc, ["en", "fr"], rows, array("L", [0]), [-1]))
    _ptc2, _pc2, classes2, rows2, _row2, output2 = load_model(path)
    assert output2 == [-1] and classes2 == ["en", "fr"] and rows2 == rows


def test_uint32_widening(tmp_path):
    '''a DFA beyond the uint16 state ceiling round-trips via uint32'''
    ptc = np.zeros((1, 2), dtype=np.float32)
    pc = np.array([0.5, 0.5], dtype=np.float32)
    rows = array("L", [1 << 16] * 256)  # state id overflows uint16
    save_model(tmp_path / "m.npz.xz",
               (ptc, pc, ["en", "fr"], rows, array("L", [0]), [0]))
    _, _, _, loaded, _, _ = load_model(tmp_path / "m.npz.xz")
    assert loaded.itemsize == 4
    assert list(loaded) == list(rows)


def test_rows_canonicalized(tmp_path):
    """rows are stored sorted and distinct, with the index remapped onto them"""
    ptc = np.zeros((1, 2), dtype=np.float32)
    pc = np.array([0.5, 0.5], dtype=np.float32)
    # rows given in descending order, the second one used by two states
    rows = array("H", [2] * 256 + [1] * 256)
    path = tmp_path / "m.npz.xz"
    save_model(path, (ptc, pc, ["en", "fr"], rows, array("L", [0, 1, 1]),
                      [0, -1, -1]))
    _, _, _, rows2, row_index, output = load_model(path)
    assert list(rows2) == [1] * 256 + [2] * 256
    assert list(row_index) == [1, 0, 0]
    # a duplicate row passed in anyway is still stored once
    save_model(path, (ptc, pc, ["en", "fr"], array("H", [1] * 512),
                      array("L", [0, 1]), [0, -1]))
    _, _, _, rows3, row_index3, _ = load_model(path)
    assert len(rows3) == 256 and list(row_index3) == [0, 0]
    assert output == [0, -1, -1]


def test_unsupported_legacy_layout_rejected(tmp_path):
    """a pre-row-dedup / pre-longest-match model is refused by name, not
    with a bare KeyError"""
    import io
    import lzma as _lzma

    import pytest

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
    path.write_bytes(_lzma.compress(buf.getvalue(), preset=1))
    with pytest.raises(ValueError, match="nextmove_row.*out_feat"):
        load_model(path)


def test_expand_nextmove():
    """the inverse of the row sharing (benchmarks/fast_eval.py's flat walk)"""
    from py3langid.modelio import expand_nextmove

    rows = array("H", [7] * 256 + [9] * 256)
    flat = expand_nextmove(rows, array("L", [1, 0, 1]))
    assert flat.typecode == "H"
    assert list(flat) == [9] * 256 + [7] * 256 + [9] * 256
    # numpy input keeps a matching typecode
    assert expand_nextmove(np.array(rows, dtype=np.uint32),
                           np.array([0, 1])).typecode == "I"
