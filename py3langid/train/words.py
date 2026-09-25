"""Token table: PMI credits for words, a fixed credit for CJK marker characters
(PMI on dense CJK characters biases whole classes)."""

import math
from collections import Counter, defaultdict

import numpy as np

from ..modelio import WordTable
from .shards import load_tokens

WORD_MIN_COUNT = 5  # keep a word if some class has at least this many occurrences
MARKER_CLASSES = ("zh", "zht", "yue", "wuu", "ja")
MARKER_RATIO = 20  # owner's doc rate vs every other marker class
MARKER_MIN_DF = 5
MARKER_MIN_RATE = 0.01  # owner doc rate; a zero runner-up alone must not qualify
MARKER_WEIGHT = 10.0  # log-score credit per distinct marker character
PMI_WEIGHT = 4.0  # multiplier on word PMI credits


def _markers(df, ndocs, lang_index):
    """{char: column} for characters near-exclusive to one marker class."""
    classes = [c for c in MARKER_CLASSES if c in lang_index]
    if len(classes) < 2:
        return {}
    out = {}
    for ch in set().union(*(df[c] for c in classes)):
        rates = np.array([df[c][ch] / ndocs[c] for c in classes])
        owner = int(rates.argmax())
        if (df[classes[owner]][ch] >= MARKER_MIN_DF and rates[owner] >= MARKER_MIN_RATE
                and rates[owner] >= MARKER_RATIO * np.partition(rates, -2)[-2]):
            out[ch] = lang_index[classes[owner]]
    return out


def build_words(shard_items, lang_index):
    """WordTable of positive credits."""
    counts, df, ndocs = defaultdict(Counter), defaultdict(Counter), Counter()
    for _, lang, shard_path in shard_items:
        words, chars, n = load_tokens(shard_path)
        counts[lang].update(words)
        df[lang].update(chars)
        ndocs[lang] += n
    markers = _markers(df, ndocs, lang_index)
    n_tok = {lang: sum(c.values()) for lang, c in counts.items()}
    n_all = sum(n_tok.values())
    total, keep = Counter(), set(markers)
    for c in counts.values():
        total.update(c)
        keep.update(w for w, n in c.items() if n >= WORD_MIN_COUNT)
    vocab = sorted(keep)
    pos = {w: i for i, w in enumerate(vocab)}
    rows = [[] for _ in vocab]  # (class column, credit) per token
    for ch, col in markers.items():
        rows[pos[ch]].append((col, MARKER_WEIGHT))
    for lang in sorted(counts, key=lang_index.get):  # deterministic CSR order
        c, col, denom = counts[lang], lang_index[lang], n_tok[lang] + 0.5 * len(vocab)
        for w, n in c.items():
            i = pos.get(w)
            if i is not None:
                credit = PMI_WEIGHT * (math.log((n + 0.5) / denom) - math.log(total[w] / n_all))
                if credit > 0:
                    rows[i].append((col, credit))
    indptr = np.cumsum([0] + [len(r) for r in rows], dtype=np.int32)
    flat = [e for r in rows for e in r]
    return WordTable("\n".join(vocab).encode(), indptr,
                     np.array([c for c, _ in flat], dtype=np.int32),
                     np.array([v for _, v in flat], dtype=np.float32))
