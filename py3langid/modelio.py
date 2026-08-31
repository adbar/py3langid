"""Model serialization: uncompressed NumPy .npz inside an LZMA stream, no pickle.
Arrays: ptc float16, pc float32, classes unicode, nextmove uint16
(auto-widened to uint32 past 65,536 DFA states), out_feat int32 = the one
feature each DFA state emits (-1 = none; the scanner emits the longest
match per position, so there is exactly one).
The DFA table is row-deduplicated: `nextmove` holds only the distinct
256-byte transition rows and `nextmove_row` maps state -> row (absent in
older models = identity).
Read-only compatibility: models written before either change carry
out_offsets/out_flat (a CSR of multi-match output) and no nextmove_row;
arrays of the retired gated blend, if present, are ignored.

The loader streams the LZMA into a temp file and reads one array at a time,
so peak memory is the largest single array rather than the whole (~90 MB)
uncompressed npz.
"""

import io
import lzma
import os
import shutil
import tempfile
from array import array

import numpy as np


def _dedup_rows(nextmove):
    """Split a flat DFA table into distinct 256-entry rows plus a state ->
    row index, narrowed to uint16 while the rows fit.
    @returns (rows flat, row index)"""
    rows = np.asarray(nextmove).reshape(-1, 256)
    uniq, index = np.unique(rows, axis=0, return_inverse=True)
    dtype = np.uint16 if len(uniq) < 1 << 16 else np.uint32
    return uniq.ravel(), index.astype(dtype).ravel()


def expand_nextmove(rows, row_index):
    """Undo the row deduplication: one 256-entry transition row per state.
    @returns an array of the same typecode as `rows`"""
    table = np.asarray(rows).reshape(-1, 256)[np.asarray(row_index)]
    return array(rows.typecode if isinstance(rows, array) else
                 {2: "H", 4: "I", 8: "L"}[table.dtype.itemsize], table.ravel())


def save_model(path, model):
    """Write a (nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output) tuple;
    tk_output = one feature index per DFA state (-1 = none)."""
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model
    nextmove = np.asarray(tk_nextmove)
    dtype = np.uint16 if not nextmove.size or nextmove.max() < 1 << 16 else np.uint32
    rows, row_index = _dedup_rows(nextmove)
    out_feat = np.asarray(tk_output, dtype=np.int32)
    assert len(out_feat) == len(row_index), "one output slot per DFA state"
    arrays = {
        # float16 log P(f|c): 0.007 nats of rounding, measured label-neutral,
        # and matmul promotes it to float32 exactly at scoring time
        "ptc": np.asarray(nb_ptc, dtype=np.float16).reshape(-1, len(nb_pc)),
        "pc": np.asarray(nb_pc, dtype=np.float32),
        "classes": np.array(nb_classes),
        "nextmove": rows.astype(dtype),
        "nextmove_row": row_index,
        "out_feat": out_feat,
    }
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with open(path, "wb") as f:
        f.write(lzma.compress(buffer.getvalue(), preset=6))


def _to_array(arr):
    """Copy an unsigned integer array into an array('H'/'I'/'L') without an
    intermediate bytes object."""
    out = array({2: "H", 4: "I", 8: "L"}[arr.dtype.itemsize])
    out.frombytes(memoryview(np.ascontiguousarray(arr)).cast("B"))
    return out


def _read_csr(offsets, flat, num_states):
    """Multi-match output of a pre-longest-match model.
    @returns a dense per-state list of feature tuples (None = none), sized
    for every DFA state as trailing states may carry no output."""
    tk_output = [None] * max(num_states, len(offsets) - 1)
    bounds = offsets.tolist()
    feats = flat.tolist()
    for state in range(len(bounds) - 1):
        lo, hi = bounds[state], bounds[state + 1]
        if hi > lo:
            tk_output[state] = tuple(feats[lo:hi])
    return tk_output


def load_model(path):
    """@returns (nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output);
    tk_nextmove holds the distinct DFA rows and tk_row maps state -> row;
    tk_output = one feature index per state (-1 = none), or, for a
    pre-longest-match model, a list of feature tuples"""
    # stream the LZMA out to a temp file so the (uncompressed) npz never
    # has to be resident, then take one array at a time
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        with lzma.open(path) as src:
            shutil.copyfileobj(src, tmp, length=1 << 20)
        tmp_path = tmp.name
    try:
        with np.load(tmp_path, allow_pickle=False) as data:
            nb_classes = data["classes"].tolist()
            tk_nextmove = _to_array(data["nextmove"])
            if "nextmove_row" in data:
                tk_row = _to_array(data["nextmove_row"])
            else:  # older models store one row per state
                tk_row = array("I", range(len(tk_nextmove) // 256))
            if "out_feat" in data:
                tk_output = data["out_feat"].tolist()
            else:  # pre-longest-match model: a CSR of multi-match output
                tk_output = _read_csr(data["out_offsets"], data["out_flat"],
                                      len(tk_row))
            return (data["ptc"], data["pc"], nb_classes, tk_nextmove, tk_row,
                    tk_output)
    finally:
        os.unlink(tmp_path)
