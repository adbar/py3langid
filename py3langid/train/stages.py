"""
stages.py -
Pure functions for the training pipeline stages: corpus indexing,
DF and LD feature selection, IG weighting, NB parameter estimation.
"""

import heapq
from collections import defaultdict

import numpy as np

from .common import (
    DF_TOKENS,
    SELECT_ORDERS,
    TOKENIZE_ORDER,
    MapPool,
    is_cjk_bigram,
    job_chunks,
    read_doc,
    set_shared,
    shared,
    walk_corpus,
)


def index_corpus(root):
    """Index a corpus/<domain>/<lang>/<doc> tree.

    @returns (items, langs, domains): (domain, lang, path) triples plus the
        class names in FIRST-APPEARANCE order along the sorted walk -- the
        column order of every matrix and of nb_classes. Deterministic, but
        NOT alphabetical. Only stability within a run matters: nb_classes
        ships with the parameters it indexes.
    """
    items = []
    langs, domains = {}, {}
    for domain, lang, path in walk_corpus(root):
        langs.setdefault(lang, None)
        domains.setdefault(domain, None)
        items.append((domain, lang, path))
    return items, list(langs), list(domains)


def ngram_select(doc_count, tokens_per_order=DF_TOKENS, orders=SELECT_ORDERS):
    """
    Top tokens_per_order terms by document frequency at each order in
    `orders`; ties break on the term itself for determinism.

    Order TOKENIZE_ORDER is restricted to CJK codepoint bigrams. Which
    orders are admissible, and that rule, are decided here and only here --
    shards may carry other terms, so order filtering belongs at selection.
    """
    buckets = defaultdict(list)
    for term, count in doc_count.items():
        order = len(term)
        if order in orders and (
                order != TOKENIZE_ORDER or is_cjk_bigram(term)):
            buckets[order].append((count, term))
    features = set()
    for bucket in buckets.values():
        top = heapq.nsmallest(tokens_per_order, bucket,
                              key=lambda x: (-x[0], x[1]))
        features.update(term for _, term in top)
    return sorted(features)


def _xlogx(v):
    """v * log(v), taking 0 * log(0) as 0."""
    log = np.zeros(v.shape, dtype=float)
    np.log(v, where=v > 0, out=log)
    return v * log


def entropy(v, axis=-1):
    """Entropy (nats) of count vectors along `axis`; an all-zero vector is 0."""
    v = np.asarray(v, dtype=float)
    total = v.sum(axis)
    nonzero = total > 0
    safe = np.where(nonzero, total, 1.0)
    return np.where(nonzero, np.log(safe) - _xlogx(v).sum(axis) / safe, 0.0)


def _binary_entropy(a, b):
    """Entropy of the two-outcome counts (a, b), broadcast elementwise."""
    total = a + b
    nonzero = total > 0
    safe = np.where(nonzero, total, 1.0)
    return np.where(nonzero,
                    np.log(safe) - (_xlogx(a) + _xlogx(b)) / safe, 0.0)


# Both IG functions below evaluate
#   IG = H(event) - P(term) H(event|term) - P(!term) H(event|!term)
# directly on the counts (`n` docs, `dist[j]` in event j, `t[i]` containing
# term i), so no contingency table is built and temporaries stay 2-D.


def compute_IG(cm_pos, dist):
    """
    Information gain per term with respect to the whole event set.

    @param cm_pos (num_term, num_event) counts of docs containing each term
    @param dist per-event document totals
    @returns (num_term,) IG values
    """
    present = np.asarray(cm_pos, dtype=float)
    dist = np.asarray(dist, dtype=float)
    n = dist.sum()
    t = present.sum(1)
    return entropy(dist) - (t * entropy(present)
                            + (n - t) * entropy(dist - present)) / n


def compute_IG_binarized(cm_pos, dist, chunk=8192):
    """
    Information gain per term, binarized with respect to each event
    (event j against the rest), chunked over terms to bound temporaries.

    @param cm_pos (num_term, num_event) counts of docs containing each term
    @param dist per-event document totals
    @returns (num_term, num_event) IG values
    """
    dist = np.asarray(dist, dtype=float)
    n = dist.sum()
    prior = _binary_entropy(dist, n - dist)
    ig = np.empty(cm_pos.shape, dtype=float)
    for lo in range(0, len(cm_pos), chunk):
        present = np.asarray(cm_pos[lo:lo + chunk], dtype=float)
        t = present.sum(1)[:, None]
        absent = dist - present
        ig[lo:lo + chunk] = prior - (
            t * _binary_entropy(present, t - present)
            + (n - t) * _binary_entropy(absent, (n - t) - absent)) / n
    return ig


def select_LD_features(ld, feats_per_lang, present):
    """
    Top feats_per_lang features per language by LD weight (IG_lang - IG_domain).
    @param ld (num_term, num_lang) LD weight matrix
    @param present (num_term, num_lang) bool; restrict each language's pick
        to features that occur in it (else the global DF pool starves
        minority-script languages and pads their quota with noise)
    @returns the union set of term row indices over languages
    """
    selected = set()
    for j, lang_w in enumerate(ld.T):
        cand = np.flatnonzero(present[:, j])
        selected.update(cand[np.argsort(lang_w[cand])[-feats_per_lang:]].tolist())
    return selected


_JUNK_BYTES = frozenset(
    b"0123456789 \t\n\r\x0b\x0c"
    b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def cluster_features(cm_lang, lang_dist, lang_index, feats, base, clusters, k):
    """Top-k *new* features per confusable cluster by cluster-restricted IG.

    Candidates are the whole feature pool, so a cluster picks whatever
    discriminates its own languages. Skipped: already-selected features (the
    budget goes to evidence the per-language quota missed), digit/punctuation
    -only candidates, and clusters with a language absent from the corpus.
    @returns set of row indices to add"""
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
    """@param chunk (lang column, path) pairs
    @returns a (n_feats, num_langs) int64 partial"""
    nm, rowbase, out, n_feats, num_langs, doc_cap = shared()
    counts = np.zeros((n_feats, num_langs), dtype=np.int64)
    for col, path in chunk:
        # the runtime's walk (langid._raw_score)
        state, visits = 0, {}
        for letter in read_doc(path, doc_cap):
            state = nm[rowbase[state] + letter]
            f = out[state]
            if f >= 0:
                visits[f] = visits.get(f, 0) + 1
        if visits:
            counts[list(visits), col] += np.fromiter(
                visits.values(), dtype=np.int64, count=len(visits))
    return counts


def feature_counts(items, tk_nextmove, tk_row, tk_output, n_feats, lang_index,
                   doc_cap, jobs=None):
    """Per-(feature, lang) occurrence counts: walk every doc through the DFA
    as the runtime does, crediting the one feature each state emits (the NB
    numerators; shard totals count every match at every position instead).
    Integer partials per worker keep the sum exact under any scheduling.
    @returns (n_feats, num_langs) int64
    """
    tasks = [(lang_index[lang], path) for _, lang, path in items]
    counts = np.zeros((n_feats, len(lang_index)), dtype=np.int64)
    rowbase = [r << 8 for r in tk_row]
    with MapPool(jobs, set_shared,
                 (tk_nextmove, rowbase, tk_output, n_feats, len(lang_index),
                  doc_cap)) as f:
        for partial in f(_feature_counts_chunk, job_chunks(tasks, jobs)):
            counts += partial
    return counts
