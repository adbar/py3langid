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


def entropy(v, axis=0):
    """
    Optimized implementation of entropy. This version is faster than that in
    scipy.stats.distributions, particularly over long vectors.
    """
    v = np.array(v, dtype='float')
    s = np.sum(v, axis=axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        rhs = np.nansum(v * np.log(v), axis=axis) / s
        r = np.log(s) - rhs
    # Where dealing with binarized events, it is possible that an event always
    # occurs and thus has 0 information. In this case, the negative class
    # will have frequency 0, resulting in log(0) being computed as nan.
    # We replace these nans with 0
    nan_index = np.isnan(rhs)
    if nan_index.any():
        r[nan_index] = 0
    return r


def _present_absent(cm_pos, dist):
    """@returns (num_term, num_event, 2) term-absent / term-present counts."""
    return np.dstack((dist - cm_pos, cm_pos))


def compute_IG(cm_pos, dist):
    """
    Information gain per term with respect to the whole event set.

    @param cm_pos (num_term, num_event) counts of docs containing each term
    @param dist per-event document totals
    @returns (num_term,) IG values
    """
    cm = _present_absent(cm_pos, dist)
    x = cm.sum(axis=1)
    term_w = x / x.sum(axis=1)[:, None].astype(float)
    # Entropy of the term-present/term-absent events
    e = entropy(cm, axis=1)
    return entropy(dist) - (term_w * e).sum(axis=1)


def compute_IG_binarized(cm_pos, dist, chunk=16):
    """
    Information gain per term, binarized with respect to each event and
    vectorized in chunks of events to bound the temp array size.

    @param cm_pos (num_term, num_event) counts of docs containing each term
    @param dist per-event document totals
    @returns (num_term, num_event) IG values
    """
    cm = _present_absent(cm_pos, dist)
    num_doc = dist.sum()
    tot = cm.sum(axis=1)
    ig = []
    for lo in range(0, cm_pos.shape[1], chunk):
        ev = cm[:, lo:lo + chunk, :]
        # (term, event, p(term), p(lang|term))
        cm_bin = np.stack((tot[:, None, :] - ev, ev), axis=2)

        e = entropy(cm_bin, axis=2)
        x = cm_bin.sum(axis=2)
        term_w = x / x.sum(axis=2)[..., None].astype(float)

        prior = np.stack((num_doc - dist[lo:lo + chunk], dist[lo:lo + chunk]),
                         axis=1).astype(float) / num_doc
        ig.append(entropy(prior.T)[None, :] - (term_w * e).sum(axis=2))
    return np.hstack(ig)


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


def is_cjk_bigram(term):
    """True for a 6-byte n-gram encoding exactly two CJK codepoints."""
    if len(term) != 6:
        return False
    try:
        s = term.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return len(s) == 2 and all(ord(ch) >= 0x2E80 for ch in s)



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


def state_visit_counts(items, tk_nextmove, num_states, lang_index,
                       doc_cap, chunk=8000):
    """Per-(DFA state, lang) visit counts over the corpus (one visit per
    byte); visits to states >= num_states are dropped, matching runtime
    scoring. @returns (num_states, num_langs) int64 counts"""
    num_langs = len(lang_index)
    lang_row = np.array([lang_index[lang] for _, lang, _ in items])
    nm = np.asarray(tk_nextmove, dtype=np.int32)
    counts = np.zeros((num_states + 1) * num_langs, dtype=np.int64)
    for lo in range(0, len(items), chunk):
        batch = items[lo:lo + chunk]
        docs = [read_doc(path, doc_cap) for _, _, path in batch]
        lens = np.array([len(b) for b in docs], dtype=np.int32)
        maxlen = int(lens.max())
        B = np.zeros((len(docs), maxlen), dtype=np.uint8)
        for i, b in enumerate(docs):
            B[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
        # sentinel row num_states collects padding and out-of-table states
        ST = np.full((len(docs), maxlen), num_states, dtype=np.int32)
        s = np.zeros(len(docs), dtype=np.int32)
        for p in range(maxlen):
            active = lens > p
            s = np.where(active, nm[((s << 8) | B[:, p])], 0)
            ST[active, p] = np.minimum(s[active], num_states)
        rows = np.asarray(lang_row[lo:lo + chunk], dtype=np.int64)
        keys = ST * num_langs + rows[:, None]
        counts += np.bincount(keys.ravel(), minlength=len(counts))
    return counts.reshape(num_states + 1, num_langs)[:num_states]


def feature_counts(state_counts, tk_output, n_feats):
    """Per-(feature, lang) occurrence counts under longest-match emission:
    each visit to a state credits the one feature that state emits.

    These are the NB numerators the runtime actually accumulates, unlike
    shard n-gram totals, which count every match at every position.
    @returns (n_feats, num_langs) int64
    """
    out = np.asarray(tk_output)[:state_counts.shape[0]]
    prod = np.zeros((n_feats, state_counts.shape[1]), dtype=np.int64)
    emitting = np.flatnonzero(out >= 0)
    np.add.at(prod, out[emitting], state_counts[emitting])
    return prod
