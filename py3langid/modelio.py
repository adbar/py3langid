"""Model serialization: uncompressed NumPy .npz inside an LZMA stream, no pickle.
Arrays: ptc float16, pc float32, classes unicode, nextmove uint16
(auto-widened to uint32 past 65,536 DFA states), out_feat int32 = the one
feature each DFA state emits (-1 = none; the scanner emits the longest
match per position, so there is exactly one).
The DFA table shares rows: `nextmove` holds only the distinct 256-byte
transition rows and `nextmove_row` maps state -> row -- as build_scanner
produces it and the runtime walks it, so save_model and load_model are
inverses. Both keys are required; models predating them are rejected.
Arrays of the retired gated blend, if present, are ignored.

The loader streams the LZMA into a temp file and reads one array at a time,
so peak memory is the largest single array rather than the whole (~90 MB)
uncompressed npz.
"""

import io
import lzma
import shutil
import tempfile
from array import array

import numpy as np


def _canonical_rows(rows, row_index):
    """Sort the transition rows and remap the state -> row index onto them,
    narrowed to uint16 while it fits. build_scanner already shares rows, so
    only the order matters here: the runtime ignores it, fixing it keeps the
    file reproducible, and sorting groups similar rows, worth ~3% to LZMA.
    @returns (rows flat, row index)"""
    uniq, index = np.unique(np.asarray(rows).reshape(-1, 256), axis=0,
                            return_inverse=True)
    dtype = np.uint16 if len(uniq) < 1 << 16 else np.uint32
    return uniq.ravel(), index.ravel()[np.asarray(row_index)].astype(dtype)


def expand_nextmove(rows, row_index):
    """Undo the row sharing: one 256-entry transition row per state. Only for
    consumers wanting a flat table; neither the file nor the runtime needs one.
    @returns an array of the same typecode as `rows`"""
    table = np.asarray(rows).reshape(-1, 256)[np.asarray(row_index)]
    return array(rows.typecode if isinstance(rows, array) else
                 {2: "H", 4: "I", 8: "L"}[table.dtype.itemsize], table.ravel())


def save_model(path, model):
    """Write a (nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output)
    tuple: build_scanner's DFA, and load_model's return value."""
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output = model
    nextmove = np.asarray(tk_nextmove)
    dtype = np.uint16 if not nextmove.size or nextmove.max() < 1 << 16 else np.uint32
    rows, row_index = _canonical_rows(nextmove, tk_row)
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


def load_model(path):
    """@returns (nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output);
    tk_nextmove holds the distinct DFA rows and tk_row maps state -> row;
    tk_output = one feature index per state (-1 = none)"""
    # stream the LZMA out to a temp file so the (uncompressed) npz never has to
    # be resident, then take one array at a time. TemporaryFile leaves nothing
    # behind even if the process is killed outright (a SIGTERM never unwinds
    # `finally`) and, unlike dropping an open fd's name, works on Windows
    with tempfile.TemporaryFile(suffix=".npz") as tmp:
        with lzma.open(path) as src:
            shutil.copyfileobj(src, tmp, length=1 << 20)
        tmp.seek(0)
        with np.load(tmp, allow_pickle=False) as data:
            missing = {"nextmove_row", "out_feat"}.difference(data.files)
            if missing:
                raise ValueError(
                    f"{path}: unsupported model layout, missing "
                    f"{sorted(missing)}; retrain with py3langid.train.train")
            return (data["ptc"], data["pc"], data["classes"].tolist(),
                    _to_array(data["nextmove"]), _to_array(data["nextmove_row"]),
                    data["out_feat"].tolist())
