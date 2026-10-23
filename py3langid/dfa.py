"""Aho-Corasick DFA over byte features: longest-match scanner, built one trie level at a time."""

from array import array
from collections import Counter
from typing import Any, NamedTuple

import numpy as np

MAX_FEATURE_BYTES = 8  # a prefix key packs one byte class per 8 bits of a uint64


class DFA(NamedTuple):
    table: bytes    # byte -> class, for bytes.translate
    nextmove: Any   # one row of transitions per state with own edges, flat
    rowbase: list   # state -> offset of its row
    output: list    # state -> longest feature ending there, or -1


def byte_classes(features):
    """(translate table, class count). A byte used by a feature is its own class: every
    state reached on it ends in that byte. Unused bytes share one class, always the root."""
    used = sorted(set(b"".join(features)))
    table = bytearray([len(used) % 256]) * 256
    for c, b in enumerate(used):
        table[b] = c
    return bytes(table), len(used) + (len(used) < 256)


def _to_array(arr):
    """stdlib array, uint16 when the values fit, else uint32."""
    code = "H" if arr.max(initial=0) < 1 << 16 else "I"
    return array(code, arr.astype(code).tobytes())


def build_dfa(features):
    """DFA emitting, at each byte, the index of the longest feature ending there."""
    table, width = byte_classes(features)
    lens = np.fromiter(map(len, features), dtype=np.int64, count=len(features))
    depth = int(lens.max(initial=0))
    if depth > MAX_FEATURE_BYTES:
        raise ValueError(f"features longer than {MAX_FEATURE_BYTES} bytes")
    lut = np.frombuffer(table, dtype=np.uint8).astype(np.uint64)
    buf = np.zeros((len(features), depth), dtype=np.uint64)
    buf[np.arange(depth) < lens[:, None]] = lut[np.frombuffer(b"".join(features), dtype=np.uint8)]
    # states numbered level by level: base[L] is the first state of depth L
    keys, base = [np.zeros(1, np.uint64)], [0, 1]
    parent, label, feat = [np.zeros(1, np.int64)], [np.zeros(1, np.int64)], [np.full(1, -1)]
    run = np.zeros(len(features), dtype=np.uint64)  # prefix key of every feature
    for L in range(1, depth + 1):
        m = lens >= L
        run = (run << np.uint64(8)) | buf[:, L - 1]
        uk, inv = np.unique(run[m], return_inverse=True)
        keys.append(uk)
        parent.append(np.searchsorted(keys[L - 1], uk >> np.uint64(8)) + base[L - 1])
        label.append((uk & np.uint64(255)).astype(np.int64))
        own = np.full(len(uk), -1, dtype=np.int64)
        ends = np.flatnonzero(lens[m] == L)
        own[inv[ends]] = np.flatnonzero(m)[ends]
        feat.append(own)
        base.append(base[L] + len(uk))
    parent, label, out = map(np.concatenate, (parent, label, feat))
    has_child = np.zeros(base[-1], dtype=bool)
    has_child[parent[1:]] = True
    has_child[0] = True  # the root owns row 0
    rowid = np.full(base[-1], -1, dtype=np.int64)
    owners = np.flatnonzero(has_child)
    rowid[owners] = np.arange(len(owners))
    rows = np.zeros((len(owners), width), dtype=np.uint32)
    fail = np.zeros(base[-1], dtype=np.int64)
    for L in range(depth + 1):
        s = np.arange(base[L], base[L + 1])
        if L > 1:  # the row of fail[parent] is complete: its depth is at most L - 2
            fail[s] = rows[rowid[fail[parent[s]]], label[s]]
        out[s] = np.where(out[s] >= 0, out[s], out[fail[s]])
        leaf = s[~has_child[s]]
        rowid[leaf] = rowid[fail[leaf]]
        inner = s[has_child[s]]
        rows[rowid[inner]] = rows[rowid[fail[inner]]]
        if L < depth:
            kids = np.arange(base[L + 1], base[L + 2])
            rows[rowid[parent[kids]], label[kids]] = kids
    return DFA(table, _to_array(rows.ravel()), (rowid * width).tolist(), out.tolist())


def visit_counts(dfa, text):
    """Longest-match feature counts over bytes, for inference and training."""
    nm, rowbase, out = dfa.nextmove, dfa.rowbase, dfa.output
    state = 0
    counts = Counter([out[state := nm[rowbase[state] + letter]]
                      for letter in text.translate(dfa.table)])
    counts.pop(-1, None)  # no feature ends there
    return counts
