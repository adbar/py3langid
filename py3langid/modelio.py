"""Model serialization: npz inside LZMA, no pickle.

NB counts are stored sparse, column-major, with the feature list: the scanner
is rebuilt at load (`dfa.build_dfa`). Other layouts are rejected.
"""

import io
import lzma
import shutil
import tempfile
from typing import Any, NamedTuple

import numpy as np

CREDIT_LEVELS = 63  # log-spaced word credit levels


class WordTable(NamedTuple):
    """CSR token credits: row i of vocab holds (cols, vals) in indptr[i]:indptr[i+1]."""
    vocab: bytes    # newline-joined tokens
    indptr: Any
    cols: Any       # class columns
    vals: Any       # credits


class Counts(NamedTuple):
    """Nonzero NB counts, column-major: (feature f, class c) is at c x len(features) + f."""
    index: Any
    values: Any


def sparse_counts(dense):
    """Counts of a (features, classes) matrix."""
    flat = np.asarray(dense).T.ravel()
    index = np.flatnonzero(flat)
    return Counts(index, flat[index])


class Model(NamedTuple):
    counts: Counts  # NB counts, COUNT_FLOOR applied
    pc: Any         # log priors
    classes: list   # column labels; aliases repeat
    features: list  # byte n-grams, one per counts row
    words: WordTable


def _escaped(a):
    """uint8 codes, 255 marks the next value of the uint32 escape list."""
    return np.minimum(a, 255).astype(np.uint8), a[a >= 255].astype(np.uint32)


def _unescaped(codes, escapes):
    a = codes.astype(np.int64)
    a[codes == 255] = escapes
    return a


def save_model(path, model):
    n_classes = len(model.pc)
    index = np.asarray(model.counts.index)
    if index.size and index[-1] >= len(model.features) * n_classes:
        raise ValueError("counts beyond the features x classes matrix")
    gap, gap_x = _escaped(np.diff(index, prepend=-1))  # column-major: zero runs are long
    val, val_x = _escaped(np.asarray(model.counts.values))
    arrays = {
        "pc": np.asarray(model.pc, dtype=np.float32),
        "classes": np.array(model.classes),
        "nb_gap": gap, "nb_gap_x": gap_x, "nb_val": val, "nb_val_x": val_x,
        "feat_bytes": np.frombuffer(b"".join(model.features), dtype=np.uint8),
        "feat_len": np.array([len(f) for f in model.features], dtype=np.uint8),
    }
    words = model.words
    vals = np.asarray(words.vals, dtype=np.float64)
    if vals.size:  # finer levels only cost bytes
        step = np.log1p(vals.max()) / CREDIT_LEVELS
        vals = np.expm1(np.round(np.log1p(vals) / step) * step)
    scale = float(vals.max()) / 255 if vals.size else 1.0
    q = np.round(vals / scale).astype(np.uint8)  # 8-bit credits
    keep = q > 0  # zero credits score nothing, rows stay so lookups still match
    arrays["wt_vocab"] = np.frombuffer(words.vocab, dtype=np.uint8)
    arrays["wt_indptr"] = np.concatenate(([0], np.cumsum(keep)))[np.asarray(words.indptr)].astype(np.int32)
    arrays["wt_cols"] = np.asarray(words.cols, dtype=np.uint8 if n_classes <= 256 else np.int32)[keep]
    arrays["wt_vals"] = q[keep]
    arrays["wt_scale"] = np.array([scale], dtype=np.float32)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with open(path, "wb") as f:
        f.write(lzma.compress(buffer.getvalue(), preset=6))


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
            missing = {"nb_gap", "feat_bytes", "wt_vocab"}.difference(data.files)
            if missing:
                raise ValueError(
                    f"{path}: unsupported model layout, missing "
                    f"{sorted(missing)}; retrain with py3langid.train.train")
            words = WordTable(data["wt_vocab"].tobytes(), data["wt_indptr"], data["wt_cols"],
                              data["wt_vals"].astype(np.float32) * data["wt_scale"][0])
            lens, blob = data["feat_len"].tolist(), data["feat_bytes"].tobytes()
            ends = np.cumsum(lens, dtype=np.int64).tolist()
            features = [blob[e - n:e] for e, n in zip(ends, lens)]
            counts = Counts(np.cumsum(_unescaped(data["nb_gap"], data["nb_gap_x"])) - 1,
                            _unescaped(data["nb_val"], data["nb_val_x"]))
            return Model(counts, data["pc"], data["classes"].tolist(), features, words)
