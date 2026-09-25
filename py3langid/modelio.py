"""Model serialization: npz inside LZMA, no pickle.

DFA rows are deduplicated: `nextmove` holds distinct 256-byte rows,
`nextmove_row` maps state → row. Both keys required; legacy models rejected.
"""

import io
import lzma
import shutil
import tempfile
from array import array
from typing import Any, NamedTuple

import numpy as np


class WordTable(NamedTuple):
    """CSR token credits: row i of vocab holds (cols, vals) in indptr[i]:indptr[i+1]."""
    vocab: bytes    # newline-joined tokens
    indptr: Any
    cols: Any       # class columns
    vals: Any       # credits


class Model(NamedTuple):
    ptc: Any        # (features, classes) NB log-probabilities
    pc: Any         # log priors
    classes: list   # column labels; aliases repeat
    nextmove: Any   # distinct 256-entry DFA rows, flat
    row: Any        # state → row
    output: list    # state → feature or -1
    words: WordTable


def _narrow(arr):
    """Unsigned ints as uint16 when they fit, else uint32."""
    arr = np.asarray(arr)
    return arr.astype(np.uint16 if not arr.size or arr.max() < 1 << 16 else np.uint32)


def _canonical_rows(rows, row_index):
    """Sort transition rows for reproducibility. Returns (flat rows, state → row)."""
    uniq, index = np.unique(np.asarray(rows).reshape(-1, 256), axis=0,
                            return_inverse=True)
    return _narrow(uniq.ravel()), _narrow(index.ravel()[np.asarray(row_index)])


def save_model(path, model):
    rows, row_index = _canonical_rows(model.nextmove, model.row)
    out_feat = np.asarray(model.output, dtype=np.int32)
    if len(out_feat) != len(row_index):
        raise ValueError("one output slot per DFA state")
    n_classes = len(model.pc)
    arrays = {
        "ptc": np.asarray(model.ptc, dtype=np.float16).reshape(-1, n_classes),
        "pc": np.asarray(model.pc, dtype=np.float32),
        "classes": np.array(model.classes),
        "nextmove": rows,
        "nextmove_row": row_index,
        "out_feat": out_feat,
    }
    words = model.words
    vals = np.asarray(words.vals, dtype=np.float32)
    scale = float(vals.max()) / 255 if vals.size else 1.0
    arrays["wt_vocab"] = np.frombuffer(words.vocab, dtype=np.uint8)
    arrays["wt_indptr"] = np.asarray(words.indptr, dtype=np.int32)
    arrays["wt_cols"] = np.asarray(words.cols, dtype=np.uint8 if n_classes <= 256 else np.int32)
    arrays["wt_vals"] = np.round(vals / scale).astype(np.uint8)  # 8-bit credits
    arrays["wt_scale"] = np.array([scale], dtype=np.float32)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with open(path, "wb") as f:
        f.write(lzma.compress(buffer.getvalue(), preset=6))


def _to_array(arr):
    """_narrow output (uint16/uint32) → stdlib array('H'/'I')."""
    out = array("H" if arr.dtype.itemsize == 2 else "I")
    out.frombytes(memoryview(np.ascontiguousarray(arr)).cast("B"))
    return out


def load_model(path):
    """Load an npz+LZMA model file into a Model."""
    # stream LZMA to a temp file so the uncompressed npz is never fully resident
    with tempfile.TemporaryFile(suffix=".npz") as tmp:
        with lzma.open(path) as src:
            shutil.copyfileobj(src, tmp, length=1 << 20)
        tmp.seek(0)
        if tmp.read(4) != b"PK\x03\x04":  # 0.3.0 models are pickles
            raise ValueError(f"{path}: unsupported model layout, not an npz archive; "
                             "retrain with py3langid.train.train")
        tmp.seek(0)
        with np.load(tmp, allow_pickle=False) as data:
            missing = {"nextmove_row", "out_feat", "wt_vocab"}.difference(data.files)
            if missing:
                raise ValueError(
                    f"{path}: unsupported model layout, missing "
                    f"{sorted(missing)}; retrain with py3langid.train.train")
            words = WordTable(data["wt_vocab"].tobytes(), data["wt_indptr"], data["wt_cols"],
                              data["wt_vals"].astype(np.float32) * data["wt_scale"][0])
            return Model(data["ptc"], data["pc"], data["classes"].tolist(),
                         _to_array(data["nextmove"]), _to_array(data["nextmove_row"]),
                         data["out_feat"].tolist(), words)
