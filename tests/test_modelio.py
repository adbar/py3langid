from array import array

import numpy as np

from py3langid.modelio import load_model, save_model


def test_roundtrip(tmp_path):
    ptc = np.arange(12, dtype=np.float32).reshape(4, 3)
    pc = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    classes = ["en", "fr", "zh"]
    nextmove = array("H", range(512))
    output = {0: (1, 3), 2: (0,)}

    path = tmp_path / "model.npz.xz"
    save_model(path, (ptc, pc, classes, nextmove, output))
    ptc2, pc2, classes2, nextmove2, output2, blend2 = load_model(path)

    assert np.array_equal(ptc2, ptc) and np.array_equal(pc2, pc)
    assert classes2 == classes
    assert nextmove2 == nextmove
    # tk_output round-trips into the dense per-state list layout
    assert output2 == [(1, 3), None, (0,)]
    assert blend2 is None

    # save_model also accepts the dense-list layout and produces the same file
    save_model(path.with_suffix(".b"), (ptc, pc, classes, nextmove, output2))
    assert path.with_suffix(".b").read_bytes() == path.read_bytes()


def test_empty_tk_output(tmp_path):
    '''model with no emitting states survives the roundtrip'''
    ptc = np.zeros((0, 2), dtype=np.float32)
    pc = np.array([0.5, 0.5], dtype=np.float32)
    nextmove = array("H", range(256))
    path = tmp_path / "model.npz.xz"
    save_model(path, (ptc, pc, ["en", "fr"], nextmove, {}))
    _ptc2, _pc2, classes2, nextmove2, output2, _blend2 = load_model(path)
    assert not any(output2) and classes2 == ["en", "fr"] and nextmove2 == nextmove


def test_uint16_ceiling(tmp_path):
    '''saving a DFA beyond the uint16 state ceiling must fail loudly'''
    import pytest
    ptc = np.zeros((1, 2), dtype=np.float32)
    pc = np.array([0.5, 0.5], dtype=np.float32)
    nextmove = array("L", [1 << 16] * 256)  # state id overflows uint16
    with pytest.raises(ValueError):
        save_model(tmp_path / "m.npz.xz", (ptc, pc, ["en", "fr"], nextmove, {0: (0,)}))
