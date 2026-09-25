"""Feature selection (DF/LD/IG) and NB feature counts."""

import heapq
from collections import defaultdict

import numpy as np

from ..langid import visit_counts
from .common import DF_TOKENS, DOC_CAP, pmap_chunks, read_doc


def ngram_select(doc_count, tokens_per_order=DF_TOKENS):
    """Top tokens_per_order terms by DF at each order."""
    buckets = defaultdict(list)
    for term, count in doc_count.items():
        buckets[len(term)].append((count, term))
    features = set()
    for bucket in buckets.values():
        top = heapq.nsmallest(tokens_per_order, bucket,
                              key=lambda x: (-x[0], x[1]))
        features.update(term for _, term in top)
    return sorted(features)


def _xlogx(v):
    """v * log(v), with 0*log(0) = 0."""
    log = np.zeros(v.shape, dtype=float)
    np.log(v, where=v > 0, out=log)
    return v * log


def entropy(v, axis=-1):
    """Entropy (nats) of count vectors; all-zero → 0."""
    v = np.asarray(v, dtype=float)
    total = v.sum(axis)
    nonzero = total > 0
    safe = np.where(nonzero, total, 1.0)
    return np.where(nonzero, np.log(safe) - _xlogx(v).sum(axis) / safe, 0.0)


def _binary_entropy(a, b):
    # inlined two-column entropy: ~2x faster than entropy(np.stack([a, b]))
    total = a + b
    nonzero = total > 0
    safe = np.where(nonzero, total, 1.0)
    return np.where(nonzero,
                    np.log(safe) - (_xlogx(a) + _xlogx(b)) / safe, 0.0)


def compute_IG(cm_pos, dist):
    """Information gain per term. Returns (num_term,) array."""
    present = np.asarray(cm_pos, dtype=float)
    dist = np.asarray(dist, dtype=float)
    n = dist.sum()
    t = present.sum(1)
    return entropy(dist) - (t * entropy(present)
                            + (n - t) * entropy(dist - present)) / n


def ld_weights(cm_lang, lang_dist, domain_ig):
    """Yield per-language LD weight arrays (IG_lang − IG_domain)."""
    dist = np.asarray(lang_dist, dtype=float)
    n = dist.sum()
    prior = _binary_entropy(dist, n - dist)
    t = cm_lang.sum(1, dtype=np.int64).astype(float)
    rest = n - t
    for j, dist_j in enumerate(dist):
        pos = np.asarray(cm_lang[:, j], dtype=float)
        neg = dist_j - pos
        yield prior[j] - (t * _binary_entropy(pos, t - pos)
                          + rest * _binary_entropy(neg, rest - neg)) / n \
            - domain_ig


def select_LD_features(cm_lang, lang_dist, domain_ig, feats_per_lang):
    """Top feats_per_lang features per language by LD weight, among those present
    in the language. Returns union of row indices."""
    if feats_per_lang < 1:
        raise ValueError("feats_per_lang must be >= 1")
    selected = set()
    for j, weight in enumerate(ld_weights(cm_lang, lang_dist, domain_ig)):
        cand = np.flatnonzero(cm_lang[:, j])
        selected.update(cand[np.argsort(weight[cand])[-feats_per_lang:]].tolist())
    return selected


def _feature_counts_chunk(nm, rowbase, out, n_feats, num_langs, chunk):
    counts = np.zeros((n_feats, num_langs), dtype=np.int64)
    for col, path in chunk:
        visits = visit_counts(nm, rowbase, out, read_doc(path, DOC_CAP)[0])
        if visits:
            counts[list(visits), col] += np.fromiter(
                visits.values(), dtype=np.int64, count=len(visits))
    return counts


def feature_counts(items, tk_nextmove, tk_row, tk_output, n_feats, lang_index,
                   jobs=1):
    """Per-(feature, lang) longest-match counts via DFA walk."""
    tasks = [(lang_index[lang], path) for _, lang, path in items]
    counts = np.zeros((n_feats, len(lang_index)), dtype=np.int64)
    rowbase = [r << 8 for r in tk_row]
    for partial in pmap_chunks(_feature_counts_chunk, tasks, jobs,
                               (tk_nextmove, rowbase, tk_output, n_feats, len(lang_index))):
        counts += partial
    return counts
