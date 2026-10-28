"""Token table: PMI credits for words over document frequencies, a fixed credit for
CJK marker characters and Han pairs (PMI on dense CJK characters biases whole classes),
Han pair bigram credits between the Chinese classes and Japanese."""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from ..langid import CJK_CLASSES
from ..modelio import WordTable
from .common import Item
from .shards import load_tokens

ENTRY_MIN_DF = 2  # a class credit needs the word in this many of its docs
DOC_OFFSET = 500  # docs added to every class total: damps thin-class credits
CREDIT_FLOOR = 2.0
MARKER_RATIO = 20  # owner's doc rate vs every other marker class
MARKER_MIN_DF = 5
MARKER_MIN_RATE = 0.01  # owner doc rate; a zero runner-up alone must not qualify
MARKER_WEIGHT = 10.0  # log-score credit per distinct marker character or pair
PAIR_RATIO = 50  # Han pairs: at 20x topical pairs (上海, 香港) pass
PAIR_MIN_DF = 20
PMI_WEIGHT = 4.0  # multiplier on word PMI credits
BIGRAM_ALPHA = 1000.0  # Dirichlet prior on the class character distribution
BIGRAM_WEIGHT = 0.5
BIGRAM_MIN_COUNT = 3  # pair occurrences over the bigram classes


def _markers(
    df: Mapping[str, Counter[str]],
    ndocs: Mapping[str, int],
    classes: Sequence[str],
    lang_index: Mapping[str, int],
) -> dict[str, int]:
    """{char or Han pair: column} for those near-exclusive to one marker class."""
    out = {}
    for ch in set().union(*(df[c] for c in classes)):
        ratio, min_df = (MARKER_RATIO, MARKER_MIN_DF) if len(ch) == 1 else (PAIR_RATIO, PAIR_MIN_DF)
        rates = np.array([df[c][ch] / ndocs[c] for c in classes])
        owner = int(rates.argmax())
        if (
            df[classes[owner]][ch] >= min_df
            and rates[owner] >= MARKER_MIN_RATE
            and rates[owner] >= ratio * np.partition(rates, -2)[-2]
        ):
            out[ch] = lang_index[classes[owner]]
    # a pair adds nothing to a marker character of the same class
    return {t: c for t, c in out.items() if len(t) == 1 or c not in (out.get(t[0]), out.get(t[1]))}


def _bigrams(
    han: Mapping[str, Counter[str]], classes: Sequence[str], lang_index: Mapping[str, int]
) -> dict[str, list[tuple[int, float]]]:
    """{Han pair: [(column, credit)]}: log P(b|a,c) - log P(b|c), lowest class at 0."""
    uni = [Counter({t: n for t, n in han[c].items() if len(t) == 1}) for c in classes]
    bi = [Counter({t: n for t, n in han[c].items() if len(t) == 2}) for c in classes]
    ctx: list[Counter[str]] = [Counter() for _ in classes]
    for k, pairs in enumerate(bi):
        for ab, n in pairs.items():
            ctx[k][ab[0]] += n
    total = sum(bi, Counter())
    v_size = len(set().union(*uni)) + 1
    norm = [sum(u.values()) + v_size for u in uni]
    out = {}
    for ab, n in total.items():
        if n < BIGRAM_MIN_COUNT:
            continue
        a, b = ab[0], ab[1]
        v = []
        for k in range(len(classes)):
            p = (uni[k][b] + 1) / norm[k]
            v.append(
                math.log((bi[k][ab] + BIGRAM_ALPHA * p) / (ctx[k][a] + BIGRAM_ALPHA)) - math.log(p)
            )
        low = min(v)
        out[ab] = [
            (lang_index[c], BIGRAM_WEIGHT * (x - low)) for c, x in zip(classes, v) if x > low
        ]
    return out


def _min_df(word: str) -> int:
    """Docs a word needs in some class: long words match as often but cost more bytes."""
    n = len(word.encode())
    return 2 if n <= 8 else 4 if n <= 14 else 6


def build_words(shard_items: Iterable[Item], lang_index: Mapping[str, int]) -> WordTable:
    """WordTable of credits of at least CREDIT_FLOOR, Han bigram credits excepted."""
    word_df: defaultdict[str, Counter[str]] = defaultdict(Counter)
    df: defaultdict[str, Counter[str]] = defaultdict(Counter)
    han: defaultdict[str, Counter[str]] = defaultdict(Counter)
    ndocs: Counter[str] = Counter()
    for _, lang, shard_path in shard_items:
        words, chars, n, han_counts = load_tokens(shard_path)
        word_df[lang].update(words)
        ndocs[lang] += n
        if lang in CJK_CLASSES:
            df[lang].update(chars)
            han[lang].update(han_counts)
    classes = [c for c in CJK_CLASSES if c in lang_index]
    n_all = sum(ndocs.values())
    total: Counter[str] = Counter()
    keep: set[str] = set()
    for c in word_df.values():
        total.update(c)
        keep.update(w for w, n in c.items() if n >= _min_df(w))
    rows: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)  # token -> (column, credit)
    if len(classes) >= 2:
        for ch, col in _markers(df, ndocs, classes, lang_index).items():
            rows[ch].append((col, MARKER_WEIGHT))
        for ab, entries in _bigrams(han, classes, lang_index).items():
            rows[ab].extend(entries)
    for lang in sorted(word_df, key=lang_index.__getitem__):  # deterministic CSR order
        c, col, log_docs = word_df[lang], lang_index[lang], math.log(ndocs[lang] + DOC_OFFSET)
        for w, n in c.items():
            if w in keep and n >= ENTRY_MIN_DF:
                credit = PMI_WEIGHT * (math.log(n) - log_docs - math.log(total[w] / n_all))
                if credit >= CREDIT_FLOOR:
                    rows[w].append((col, credit))
    vocab = sorted(rows)
    indptr = np.cumsum([0] + [len(rows[w]) for w in vocab], dtype=np.int32)
    flat = [e for w in vocab for e in rows[w]]
    return WordTable(
        "\n".join(vocab).encode(),
        indptr,
        np.array([c for c, _ in flat], dtype=np.int32),
        np.array([v for _, v in flat], dtype=np.float32),
    )
