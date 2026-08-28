"""Model serialization: uncompressed NumPy .npz inside an LZMA stream, no pickle.
Arrays: ptc float32, pc float32, classes unicode, nextmove uint16,
out_offsets/out_flat uint32 (CSR encoding of tk_output).
Optional gated-blend arrays: blend_ptc float16 (num_states, num_class),
blend_cluster_id int16 (num_class, -1 = no cluster)."""

import io
import lzma
from array import array

import numpy as np


def save_model(path, model, blend=None):
    """Write a (nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output) tuple;
    tk_output = dict or dense per-state sequence; blend = optional
    (blend_ptc, blend_cluster_id) arrays."""
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model
    if not isinstance(tk_output, dict):
        tk_output = {s: v for s, v in enumerate(tk_output) if v}
    nextmove = np.asarray(tk_nextmove)
    if nextmove.size and nextmove.max() >= 1 << 16:
        raise ValueError("DFA exceeds uint16 state ceiling; nextmove would overflow")
    num_states = max(tk_output) + 1 if tk_output else 0
    lengths, flat = np.zeros(num_states + 1, dtype=np.uint32), []
    for state, feats in sorted(tk_output.items()):
        lengths[state + 1] = len(feats)
        flat.extend(feats)
    arrays = {
        "ptc": np.asarray(nb_ptc, dtype=np.float32).reshape(-1, len(nb_pc)),
        "pc": np.asarray(nb_pc, dtype=np.float32),
        "classes": np.array(nb_classes),
        "nextmove": nextmove.astype(np.uint16),
        "out_offsets": np.cumsum(lengths, dtype=np.uint32),
        "out_flat": np.asarray(flat, dtype=np.uint32),
    }
    if blend is not None:
        blend_ptc, cluster_id = blend
        arrays["blend_ptc"] = np.asarray(blend_ptc, dtype=np.float16)
        arrays["blend_cluster_id"] = np.asarray(cluster_id, dtype=np.int16)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with open(path, "wb") as f:
        f.write(lzma.compress(buffer.getvalue(), preset=6))


def load_model(path):
    """@returns (nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output, blend);
    tk_output = dense per-state list (None = no match); blend =
    (blend_ptc float32, blend_cluster_id) or None"""
    with open(path, "rb") as f:
        data = np.load(io.BytesIO(lzma.decompress(f.read())), allow_pickle=False)
    nb_classes = data["classes"].tolist()
    tk_nextmove = array("H")
    tk_nextmove.frombytes(data["nextmove"].tobytes())
    offsets, flat = data["out_offsets"].tolist(), data["out_flat"].tolist()
    tk_output = [None] * max(len(tk_nextmove) // 256, len(offsets) - 1)
    for state in range(len(offsets) - 1):
        if offsets[state + 1] > offsets[state]:
            tk_output[state] = tuple(flat[offsets[state]:offsets[state + 1]])
    blend = None
    if "blend_ptc" in data:
        blend = (data["blend_ptc"].astype(np.float32), data["blend_cluster_id"])
    return data["ptc"], data["pc"], nb_classes, tk_nextmove, tk_output, blend
