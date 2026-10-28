"""Aho-Corasick DFA over byte features: longest-match scanner, built one trie level at a time."""

from array import array
from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


class DFA(NamedTuple):
    table: bytes  # byte -> class, for bytes.translate
    nextmove: "array[int]"  # one row of transitions per state with own edges, flat
    rowbase: list[int]  # state -> offset of its row
    output: list[int]  # state -> longest feature ending there, or -1


def byte_classes(data: bytes) -> tuple[bytes, int]:
    """(translate table, class count). A byte of *data* is its own class: every state
    reached on it ends in that byte. Other bytes share one class, always the root."""
    used = np.bincount(np.frombuffer(data, np.uint8), minlength=256) > 0
    table = np.where(used, used.cumsum() - 1, used.sum()).astype(np.uint8).tobytes()
    return table, max(table) + 1


def build_dfa(features: Sequence[bytes]) -> DFA:
    """DFA emitting, at each byte, the index of the longest feature ending there."""
    joined = b"".join(features)
    table, width = byte_classes(joined)
    flat = np.frombuffer(joined.translate(table), dtype=np.uint8)
    lens = np.fromiter(map(len, features), dtype=np.int64, count=len(features))
    start = np.cumsum(lens) - lens
    # states numbered level by level: base[L] is the first state of depth L
    edges, base = [np.zeros(1, np.int64)], [0, 1]  # edge = parent * width + class
    node = np.zeros(len(features), dtype=np.int64)  # state of every feature's prefix
    for L in range(1, lens.max(initial=0) + 1):
        m = lens >= L
        edge, inv = np.unique(node[m] * width + flat[start[m] + L - 1], return_inverse=True)
        edges.append(edge)
        node[m] = base[L] + inv
        base.append(base[L] + len(edge))
    parent, label = np.divmod(np.concatenate(edges), width)
    out = np.full(base[-1], -1, dtype=np.int64)
    out[node] = np.arange(len(features))
    has_child = np.bincount(parent, minlength=base[-1]) > 0  # parent[0] is 0: the root owns row 0
    rowid = np.cumsum(has_child) - 1  # leaves take their fail's row below
    code = "H" if base[-1] <= 1 << 16 else "I"
    nextmove = array(code, [0]) * ((rowid[-1] + 1) * width)
    rows = np.frombuffer(nextmove, dtype=code).reshape(-1, width)  # writes go to nextmove
    fail = np.zeros(base[-1], dtype=np.int64)
    for L in range(len(base) - 1):
        s = np.arange(base[L], base[L + 1])
        fail[s] = rows[rowid[fail[parent[s]]], label[s]]  # read before this level's edges
        rows[rowid[parent[s]], label[s]] = s
        out[s] = np.where(out[s] >= 0, out[s], out[fail[s]])
        rowid[s] = np.where(has_child[s], rowid[s], rowid[fail[s]])
        inner = s[has_child[s]]
        rows[rowid[inner]] = rows[rowid[fail[inner]]]
    return DFA(table, nextmove, (rowid * width).tolist(), out.tolist())


def visit_counts(dfa: DFA, text: bytes) -> Counter[int]:
    """Longest-match feature counts over bytes, for inference and training."""
    nm, rowbase, out = dfa.nextmove, dfa.rowbase, dfa.output
    state = 0
    counts = Counter(
        [out[state := nm[rowbase[state] + letter]] for letter in text.translate(dfa.table)]
    )
    counts.pop(-1, None)  # no feature ends there
    return counts
