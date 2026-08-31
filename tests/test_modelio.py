from array import array

import numpy as np

from py3langid.modelio import load_model, save_model


def test_roundtrip(tmp_path):
    ptc = np.arange(12, dtype=np.float32).reshape(4, 3)
    pc = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    classes = ["en", "fr", "zh"]
    nextmove = array("H", range(512))
    output = [3, -1]  # one longest-match feature per state, -1 = none

    path = tmp_path / "model.npz.xz"
    save_model(path, (ptc, pc, classes, nextmove, output))
    ptc2, pc2, classes2, nextmove2, row2, output2 = load_model(path)

    assert np.array_equal(ptc2, ptc) and np.array_equal(pc2, pc)
    assert classes2 == classes
    assert nextmove2 == nextmove and list(row2) == [0, 1]
    assert output2 == [3, -1]

    # an array of the same values produces the same file
    save_model(path.with_suffix(".b"),
               (ptc, pc, classes, nextmove, array("l", output)))
    assert path.with_suffix(".b").read_bytes() == path.read_bytes()


def test_empty_tk_output(tmp_path):
    '''model with no emitting states survives the roundtrip'''
    ptc = np.zeros((0, 2), dtype=np.float32)
    pc = np.array([0.5, 0.5], dtype=np.float32)
    nextmove = array("H", range(256))
    path = tmp_path / "model.npz.xz"
    save_model(path, (ptc, pc, ["en", "fr"], nextmove, [-1]))
    _ptc2, _pc2, classes2, nextmove2, _row2, output2 = load_model(path)
    assert output2 == [-1] and classes2 == ["en", "fr"] and nextmove2 == nextmove


def test_uint32_widening(tmp_path):
    '''a DFA beyond the uint16 state ceiling round-trips via uint32'''
    ptc = np.zeros((1, 2), dtype=np.float32)
    pc = np.array([0.5, 0.5], dtype=np.float32)
    nextmove = array("L", [1 << 16] * 256)  # state id overflows uint16
    save_model(tmp_path / "m.npz.xz", (ptc, pc, ["en", "fr"], nextmove, [0]))
    _, _, _, loaded, _, _ = load_model(tmp_path / "m.npz.xz")
    assert loaded.itemsize == 4
    assert list(loaded) == list(nextmove)


def test_row_dedup(tmp_path):
    """identical transition rows are stored once, with a state -> row index"""
    ptc = np.zeros((1, 2), dtype=np.float32)
    pc = np.array([0.5, 0.5], dtype=np.float32)
    # three states, rows 0 and 2 identical
    nextmove = array("H", [1] * 256 + [2] * 256 + [1] * 256)
    path = tmp_path / "m.npz.xz"
    save_model(path, (ptc, pc, ["en", "fr"], nextmove, [0, -1, -1]))
    _, _, _, rows, row_index, output = load_model(path)
    assert len(rows) == 512 and list(row_index) == [0, 1, 0]
    # the expanded table is unchanged, and every state has an output slot
    expanded = np.asarray(rows).reshape(-1, 256)[np.asarray(row_index)].ravel()
    assert list(expanded) == list(nextmove) and output == [0, -1, -1]


def test_legacy_model_without_row_index(tmp_path):
    """a model saved before row dedup loads with an identity row index"""
    import io
    import lzma as _lzma

    ptc = np.zeros((1, 2), dtype=np.float32)
    arrays = {
        "ptc": ptc, "pc": np.array([0.5, 0.5], dtype=np.float32),
        "classes": np.array(["en", "fr"]),
        "nextmove": np.array([1] * 256 + [2] * 256, dtype=np.uint16),
        "out_offsets": np.array([0, 1, 1], dtype=np.uint32),
        "out_flat": np.array([0], dtype=np.uint32),
    }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    path = tmp_path / "old.npz.xz"
    path.write_bytes(_lzma.compress(buf.getvalue(), preset=1))
    _, _, _, rows, row_index, output = load_model(path)
    assert len(rows) == 512 and list(row_index) == [0, 1]
    assert output == [(0,), None]


def test_legacy_csr_output(tmp_path):
    """a pre-longest-match model (CSR multi-match output) still loads"""
    import io
    import lzma as _lzma

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
    _, _, _, rows, row_index, output = load_model(path)
    assert list(row_index) == [0, 1, 2] and len(rows) == 768
    assert output == [(1, 3), None, (0,)]
