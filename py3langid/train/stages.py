"""Corpus indexing, feature selection (DF/LD/IG), NB parameter estimation."""

import heapq
from collections import defaultdict

import numpy as np

from .common import (
    DF_TOKENS,
    DOC_CAP,
    SELECT_ORDERS,
    MapPool,
    job_chunks,
    read_doc,
    set_shared,
    shared,
    walk_corpus,
)


def index_corpus(root):
    """Returns (items, langs, domains) in first-appearance walk order."""
    items = []
    langs, domains = {}, {}
    for domain, lang, path in walk_corpus(root):
        langs.setdefault(lang, None)
        domains.setdefault(domain, None)
        items.append((domain, lang, path))
    return items, list(langs), list(domains)


def ngram_select(doc_count, tokens_per_order=DF_TOKENS, orders=SELECT_ORDERS):
    """Top tokens_per_order terms by DF at each admissible order."""
    buckets = defaultdict(list)
    for term, count in doc_count.items():
        order = len(term)
        if order in orders:
            buckets[order].append((count, term))
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


def select_LD_features(ld_columns, feats_per_lang, present):
    """Top feats_per_lang per language by LD weight. Returns union of row indices."""
    selected = set()
    for j, lang_w in enumerate(ld_columns):
        cand = np.flatnonzero(present[:, j])
        selected.update(cand[np.argsort(lang_w[cand])[-feats_per_lang:]].tolist())
    return selected


_JUNK_BYTES = frozenset(
    b"0123456789 \t\n\r\x0b\x0c"
    b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def cluster_features(cm_lang, lang_dist, lang_index, feats, base, clusters, k):
    """Top-k new features per confusable cluster by cluster-restricted IG."""
    added = set()
    selected = set(base)
    for cluster in clusters:
        if any(lang not in lang_index for lang in cluster):
            continue
        cols = [lang_index[lang] for lang in cluster]
        ig = compute_IG(cm_lang[:, cols], lang_dist[cols])
        taken = 0
        for t in np.argsort(ig)[::-1]:
            t = int(t)
            if t in selected or all(b in _JUNK_BYTES for b in feats[t]):
                continue
            added.add(t)
            selected.add(t)
            taken += 1
            if taken >= k:
                break
    return added


def _feature_counts_chunk(chunk):
    nm, rowbase, out, n_feats, num_langs = shared()
    counts = np.zeros((n_feats, num_langs), dtype=np.int64)
    for col, path in chunk:
        state, visits = 0, {}
        for letter in read_doc(path, DOC_CAP):
            state = nm[rowbase[state] + letter]
            f = out[state]
            if f >= 0:
                visits[f] = visits.get(f, 0) + 1
        if visits:
            counts[list(visits), col] += np.fromiter(
                visits.values(), dtype=np.int64, count=len(visits))
    return counts


def feature_counts(items, tk_nextmove, tk_row, tk_output, n_feats, lang_index,
                   jobs=None):
    """Per-(feature, lang) longest-match counts via DFA walk."""
    tasks = [(lang_index[lang], path) for _, lang, path in items]
    counts = np.zeros((n_feats, len(lang_index)), dtype=np.int64)
    rowbase = [r << 8 for r in tk_row]
    with MapPool(jobs, set_shared,
                 (tk_nextmove, rowbase, tk_output, n_feats,
                  len(lang_index))) as f:
        for partial in f(_feature_counts_chunk, job_chunks(tasks, jobs)):
            counts += partial
    return counts
