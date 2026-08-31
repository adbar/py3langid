"""
stages.py -
Pure functions for the training pipeline stages: corpus indexing,
DF and LD feature selection, IG weighting, NB parameter estimation.
"""

import heapq
from collections import defaultdict

import numpy as np

from .common import read_doc, walk_corpus


def index_corpus(root):
    """Index a corpus/<domain>/<lang>/<doc> tree.

    @returns (items, langs, domains): (domain, lang, path) string triples
        plus the language and domain names in sorted (walk) order --
        the class column order of every downstream matrix
    """
    items = []
    langs, domains = {}, {}
    for domain, lang, path in walk_corpus(root):
        langs.setdefault(lang, None)
        domains.setdefault(domain, None)
        items.append((domain, lang, path))
    return items, list(langs), list(domains)


def ngram_select(doc_count, max_order, tokens_per_order, min_order):
    """
    DF feature selection for byte-ngram tokenization: top tokens_per_order
    terms by document frequency for each order. Ties break on the term
    itself for determinism.
    """
    buckets = defaultdict(list)
    for term, count in doc_count.items():
        if min_order <= len(term) <= max_order:
            buckets[len(term)].append((count, term))
    features = set()
    for order in range(min_order, max_order + 1):
        top = heapq.nsmallest(tokens_per_order, buckets[order],
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


def select_LD_features(ld, feats_per_lang, present=None):
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
        cand = (np.flatnonzero(present[:, j]) if present is not None
                else np.arange(len(lang_w)))
        selected.update(cand[np.argsort(lang_w[cand])[-feats_per_lang:]].tolist())
    return selected


def learn_pc(class_counts):
    """
    @param class_counts per-class document counts
    @returns nb_pc: log(P(C))
    """
    class_counts = np.asarray(class_counts)
    assert (class_counts > 0).all(), "every language must have at least one document"
    return np.log(class_counts)


def prod_to_ptc(prod, alpha=1.0):
    """@returns nb_ptc: log(P(t|C)), (num_term, num_class), from a
    term x lang total-occurrence matrix"""
    return np.log(alpha + prod) - np.log(alpha * prod.shape[0] + prod.sum(0))


_JUNK_BYTES = frozenset(
    b"0123456789 \t\n\r\x0b\x0c"
    b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def cluster_features(cm_lang, lang_dist, lang_index, feats, base, clusters, k):
    """Top-k *new* features per confusable cluster by cluster-restricted IG.

    One mechanism for what used to be two: the candidate set is the whole
    feature pool, byte n-grams and the order-6 CJK codepoint bigrams alike,
    so a cluster picks whichever discriminates its own languages. Features
    already selected are skipped (a cluster spends its budget on evidence
    the per-language quota missed), as are digit/punctuation-only
    candidates and clusters with a language absent from the corpus.
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


def feature_counts(items, tk_nextmove, tk_row, tk_output, n_feats, lang_index,
                   doc_cap, chunk=8000):
    """Per-(feature, lang) occurrence counts: walk every doc through the DFA
    and credit the one feature each state emits. These are the NB numerators
    the runtime accumulates, unlike shard n-gram totals, which count every
    match at every position.
    @returns (n_feats, num_langs) int64
    """
    num_langs = len(lang_index)
    lang_row = np.array([lang_index[lang] for _, lang, _ in items])
    nm = np.asarray(tk_nextmove, dtype=np.int32)
    rowbase = np.asarray(tk_row, dtype=np.int32) << 8  # as the runtime walks
    num_states = len(tk_output)
    # mapped to features inside the walk, so nothing state-shaped is ever
    # materialized. Two dump slots keep it branch-free: state num_states for
    # anything off the table, feature n_feats for that plus non-emitting
    # states and a short doc's padding
    emits = np.asarray(tk_output, dtype=np.int32)
    emits = np.append(np.where(emits >= 0, emits, n_feats), n_feats)
    counts = np.zeros((n_feats + 1) * num_langs, dtype=np.int64)
    for lo in range(0, len(items), chunk):
        docs = [read_doc(path, doc_cap) for _, _, path in items[lo:lo + chunk]]
        lens = np.array([len(b) for b in docs], dtype=np.int32)
        maxlen = int(lens.max())
        B = np.zeros((len(docs), maxlen), dtype=np.uint8)
        for i, b in enumerate(docs):
            B[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
        keys = np.full((len(docs), maxlen), n_feats, dtype=np.int64)
        s = np.zeros(len(docs), dtype=np.int32)
        for p in range(maxlen):
            active = lens > p
            s = np.where(active, nm[rowbase[s] + B[:, p]], 0)
            keys[active, p] = emits[np.minimum(s[active], num_states)]
        keys *= num_langs
        keys += np.asarray(lang_row[lo:lo + chunk], dtype=np.int64)[:, None]
        counts += np.bincount(keys.ravel(), minlength=len(counts))
    return counts.reshape(n_feats + 1, num_langs)[:n_feats]
