"""Token table: PMI credits for words over document frequencies, a fixed credit for
CJK marker characters (PMI on dense CJK characters biases whole classes)."""

import math
from collections import Counter, defaultdict

import numpy as np

from ..modelio import WordTable
from .shards import TOKENS, load_shard

ENTRY_MIN_DF = 2  # a class credit needs the word in this many of its docs
DOC_OFFSET = 500  # docs added to every class total: damps thin-class credits
CREDIT_FLOOR = 2.0
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


def _min_df(word):
    """Docs a word needs in some class: long words match as often but cost more bytes."""
    n = len(word.encode())
    return 2 if n <= 8 else 4 if n <= 14 else 6


def build_words(shard_items, lang_index):
    """WordTable of credits of at least CREDIT_FLOOR."""
    word_df, df, ndocs = defaultdict(Counter), defaultdict(Counter), Counter()
    for _, lang, shard_path in shard_items:
        words, chars, n = load_shard(shard_path, TOKENS)
        word_df[lang].update(words)
        df[lang].update(chars)
        ndocs[lang] += n
    markers = _markers(df, ndocs, lang_index)
    n_all = sum(ndocs.values())
    total, keep = Counter(), set()
    for c in word_df.values():
        total.update(c)
        keep.update(w for w, n in c.items() if n >= _min_df(w))
    rows = defaultdict(list)  # token -> (class column, credit)
    for ch, col in markers.items():
        rows[ch].append((col, MARKER_WEIGHT))
    for lang in sorted(word_df, key=lang_index.get):  # deterministic CSR order
        c, col, log_docs = word_df[lang], lang_index[lang], math.log(ndocs[lang] + DOC_OFFSET)
        for w, n in c.items():
            if w in keep and n >= ENTRY_MIN_DF:
                credit = PMI_WEIGHT * (math.log(n) - log_docs - math.log(total[w] / n_all))
                if credit >= CREDIT_FLOOR:
                    rows[w].append((col, credit))
    vocab = sorted(rows)
    indptr = np.cumsum([0] + [len(rows[w]) for w in vocab], dtype=np.int32)
    flat = [e for w in vocab for e in rows[w]]
    return WordTable("\n".join(vocab).encode(), indptr,
                     np.array([c for c, _ in flat], dtype=np.int32),
                     np.array([v for _, v in flat], dtype=np.float32))
