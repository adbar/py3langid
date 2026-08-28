"""
stages.py -
Pure functions for the training pipeline stages: corpus indexing,
DF and LD feature selection, IG weighting, NB parameter estimation.
"""

import heapq
from collections import defaultdict

import numpy as np

from .common import (
    BLEND_ALPHA,
    BLEND_CLUSTERS,
    BLEND_LAMBDA,
    LABEL_ALIAS,
    NEEDY_LANGS,
    QUOTA_NEEDY,
    QUOTA_TRIMMED,
    TRIM_LANGS,
    read_doc,
    walk_corpus,
)


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


def select_quota_features(ld, present, base_mask, langs, default_quota):
    """Per-language quotas: NEEDY_LANGS draw QUOTA_NEEDY from the full
    pool, TRIM_LANGS QUOTA_TRIMMED and the rest default_quota from the
    base_mask rows. @returns the union set of term row indices"""
    selected = set()
    for j, lang in enumerate(langs):
        alias = LABEL_ALIAS.get(lang, lang)
        if alias in NEEDY_LANGS:
            quota, cand = QUOTA_NEEDY, np.flatnonzero(present[:, j])
        else:
            quota = QUOTA_TRIMMED if alias in TRIM_LANGS else default_quota
            cand = np.flatnonzero(present[:, j] & base_mask)
        order = cand[np.argsort(ld[cand, j])][::-1]
        selected.update(order[:quota].tolist())
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


def group_features(cm_lang, lang_dist, lang_index, DFfeats, base, groups, k):
    """Top-k new features per language group by group-restricted IG;
    digit/punctuation-only candidates and groups with absent languages
    are skipped. @returns set of DFfeats row indices to add"""
    added = set()
    selected = set(base)
    for group in groups:
        if any(lang not in lang_index for lang in group):
            continue
        cols = [lang_index[lang] for lang in group]
        ig = compute_IG(cm_lang[:, cols], lang_dist[cols])
        taken = 0
        for t in np.argsort(ig)[::-1]:
            t = int(t)
            if t in selected or all(b in _JUNK_BYTES for b in DFfeats[t]):
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


def blend_table(nb_ptc, tk_output, state_counts, alpha, lam):
    """Blend table S = log(lam*P(state|c) + (1-lam)*P_fold(state|c)),
    P_fold = per-state sum of log P(f|c) over output features.
    @returns (num_states, num_class) float32"""
    num_states = state_counts.shape[0]
    fold = np.zeros((num_states, nb_ptc.shape[1]))
    for state, feats in tk_output.items():
        if state < num_states:
            fold[state] = nb_ptc[list(feats)].sum(axis=0)
    C = state_counts.astype(np.float64)
    state_nb = np.log(C + alpha) - np.log(C.sum(axis=0) + alpha * num_states)
    return np.logaddexp(state_nb + np.log(lam),
                        fold + np.log1p(-lam)).astype(np.float32)


def build_blend(items, lang_index, nb_classes, nb_ptc, tk_nextmove, tk_output,
                doc_cap):
    """Gated-blend arrays: state-level blend table + per-class cluster ids.
    @returns (blend_ptc float32, cluster_id int16) for modelio.save_model"""
    num_states = max(tk_output) + 1 if tk_output else 0
    counts = state_visit_counts(items, tk_nextmove, num_states, lang_index,
                                doc_cap)
    blend_ptc = blend_table(nb_ptc, tk_output, counts, BLEND_ALPHA, BLEND_LAMBDA)
    cluster_id = np.full(len(nb_classes), -1, dtype=np.int16)
    for gi, group in enumerate(BLEND_CLUSTERS):
        for i, c in enumerate(nb_classes):
            if c in group:
                cluster_id[i] = gi
    return blend_ptc, cluster_id
