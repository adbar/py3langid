"""Feature selection (DF/LD/IG) and NB feature counts."""

import heapq
from collections import defaultdict

import numpy as np

from ..dfa import build_dfa, visit_counts
from .common import CELL_COST, COUNT_FLOOR, DF_TOKENS, DOC_CAP, pmap_chunks, read_doc


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


def _entropy(total, xlogx_sum):
    """Entropy (nats) from a total and its summed x log x, 0 for a zero total."""
    nonzero = total > 0
    safe = np.where(nonzero, total, 1.0)
    return np.where(nonzero, np.log(safe) - xlogx_sum / safe, 0.0)


def entropy(v):
    v = np.asarray(v, dtype=float)
    return _entropy(v.sum(-1), _xlogx(v).sum(-1))


def _binary_entropy(a, b):
    # two columns without stacking: ~2x faster than entropy(np.stack([a, b]))
    return _entropy(a + b, _xlogx(a) + _xlogx(b))


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
    in the language. Positive weights are discounted by the share of classes with
    DF above COUNT_FLOOR: each one costs a model cell. Returns union of row indices."""
    if feats_per_lang < 1:
        raise ValueError("feats_per_lang must be >= 1")
    share = (cm_lang > COUNT_FLOOR).sum(1) / cm_lang.shape[1]
    selected = set()
    for j, weight in enumerate(ld_weights(cm_lang, lang_dist, domain_ig)):
        cand = np.flatnonzero(cm_lang[:, j])
        w = weight[cand]
        score = np.where(w > 0, w / (1 + CELL_COST * share[cand]), w)
        selected.update(cand[np.argsort(score)[-feats_per_lang:]].tolist())
    return selected


def _feature_counts_chunk(dfa, n_feats, num_langs, chunk):
    counts = np.zeros((n_feats, num_langs), dtype=np.int64)
    for col, path in chunk:
        visits = visit_counts(dfa, read_doc(path, DOC_CAP)[0])
        if visits:
            counts[list(visits), col] += np.fromiter(
                visits.values(), dtype=np.int64, count=len(visits))
    return counts


def feature_counts(items, features, lang_index, jobs=1):
    """Per-(feature, lang) longest-match counts via DFA walk."""
    dfa = build_dfa(features)
    print(f"scanner: {len(dfa.output)} states, {sum(f >= 0 for f in dfa.output)} emitting")
    tasks = [(lang_index[lang], path) for _, lang, path in items]
    counts = np.zeros((len(features), len(lang_index)), dtype=np.int64)
    for partial in pmap_chunks(_feature_counts_chunk, tasks, jobs,
                               (dfa, len(features), len(lang_index))):
        counts += partial
    return counts
