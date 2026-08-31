"""
shards.py -
Per-(domain, lang) n-gram count shards.

One tokenization pass produces, per (domain, lang) directory, a
term -> document frequency dict (see doc_ngrams). Feature selection is
algebra over these shards, so a corpus edit only retokenizes the
directories it touches. The NB parameters come separately from
feature_counts, which credits one longest match per byte position.

Selection takes two passes: merge_docfreq tallies globally to pick the
candidate pool, count_matrices then projects the shards onto it, yielding
everything the IG stage needs.

A shard is two marshal objects, key header then docfreq payload. The key
hashes the docs' (filename, size, mtime) -- so it is location-independent
-- plus the tokenization constants, and reuse needs exact equality: no
payload is read under settings other than the ones that wrote it.

Aggregations reduce integer partials, exact under any worker scheduling.
"""

import hashlib
import marshal
import os
from collections import Counter, defaultdict

import numpy as np

from .common import (
    MAX_NGRAM_ORDER,
    MIN_NGRAM_ORDER,
    TOKENIZE_ORDER,
    MapPool,
    chunks,
    is_cjk_bigram,
    job_chunks,
    read_doc,
    set_shared,
    shared,
)

# small chunks keep each worker's partial Counter small (one chunk per job
# grew a near-global Counter in every worker: 8.7 -> 4.7 GB peak)
MERGE_SHARDS_PER_CHUNK = 8

# document frequencies, bounded by docs per class (~10^3); int32 halves the
# count matrices, which every worker allocates in full and ships back
COUNT_DTYPE = np.int32


def doc_ngrams(data, max_order):
    """Distinct terms one doc contributes: byte n-grams of order
    MIN_NGRAM_ORDER..max_order, plus the CJK codepoint bigrams at
    TOKENIZE_ORDER. That this order carries *only* CJK bigrams is enforced
    by ngram_select, not here."""
    terms = set()
    n = len(data)
    for i in range(n):
        for k in range(MIN_NGRAM_ORDER, min(max_order, n - i) + 1):
            terms.add(data[i:i + k])
        # only 3+3 bytes can pass is_cjk_bigram, and a codepoint >= U+2E80
        # has lead byte >= 0xE2: two comparisons skip almost every position
        if i + TOKENIZE_ORDER <= n \
                and data[i] >= 0xE2 and data[i + 3] >= 0xE2:
            term = data[i:i + TOKENIZE_ORDER]
            if is_cjk_bigram(term):
                terms.add(term)
    return terms


def group_items(items):
    """Group (domain, lang, path) triples by (domain, lang), sorted."""
    groups = defaultdict(list)
    for domain, lang, path in items:
        groups[(domain, lang)].append(path)
    return sorted(groups.items())


def _group_key(paths, doc_cap):
    """Cache key: the docs' (filename, size, mtime) plus everything that
    changes what doc_ngrams emits, so editing a tokenization constant
    invalidates the cache."""
    h = hashlib.sha256()
    h.update(f"{MIN_NGRAM_ORDER}\0{MAX_NGRAM_ORDER}\0{TOKENIZE_ORDER}\0"
             f"{doc_cap}\0".encode())
    for p in sorted(paths):
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return h.hexdigest()


def _build_shard(arg):
    """Build one shard unless a valid cached one exists.
    @returns (shard_path, built)
    """
    doc_cap, = shared()
    shard_path, key, paths = arg
    try:
        with open(shard_path, 'rb') as f:
            if marshal.load(f) == key:
                return shard_path, False
    except (OSError, EOFError, ValueError, TypeError):
        pass

    docfreq = Counter()
    for path in paths:
        docfreq.update(doc_ngrams(read_doc(path, doc_cap), MAX_NGRAM_ORDER))

    tmp_path = shard_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        marshal.dump(key, f)
        marshal.dump(dict(docfreq), f)
    os.replace(tmp_path, shard_path)
    return shard_path, True


def build_shards(items, shard_dir, jobs=None, doc_cap=0):
    """Build (or reuse cached) shards for all (domain, lang) groups.

    @param items (domain, lang, path) triples with string names
    @param doc_cap truncate each doc to this many bytes (0 = no cap);
        part of the shard filename, so one shard dir serves any mix of caps
    @returns list of (domain, lang, shard_path), sorted by (domain, lang)
    """
    os.makedirs(shard_dir, exist_ok=True)
    tasks = []
    shard_items = []
    for (domain, lang), paths in group_items(items):
        shard_path = os.path.join(shard_dir, f"{domain}__{lang}.cap{doc_cap}")
        tasks.append((shard_path, _group_key(paths, doc_cap), paths))
        shard_items.append((domain, lang, shard_path))

    with MapPool(jobs, set_shared, (doc_cap,)) as f:
        built = sum(new for _, new in f(_build_shard, tasks))
    print(f"shards: {built} built, {len(tasks) - built} cached")
    return shard_items


def load_shard(shard_path):
    """@returns the term -> document frequency dict of a shard."""
    with open(shard_path, 'rb') as f:
        marshal.load(f)
        return marshal.load(f)


def _merge_chunk(chunk):
    merged = Counter()
    for _, _, shard_path in chunk:
        merged.update(load_shard(shard_path))
    return merged


def merge_docfreq(shard_items, jobs=None):
    """Global term -> document frequency over all shards."""
    doc_count = Counter()
    with MapPool(jobs) as f:
        for partial in f(_merge_chunk,
                         chunks(shard_items, MERGE_SHARDS_PER_CHUNK)):
            doc_count.update(partial)
    return doc_count


def _select_counts(counts, feat_index):
    """Intersect a shard count dict with a feature index, iterating the
    smaller of the two.
    @returns (row indices, counts) with unique indices
    """
    # a shard may share no feature with the pool (e.g. a lang whose docs are
    # all empty); the arrays keep `+=` typed rather than inferring float64
    idx, vals = [], []
    if len(counts) < len(feat_index):
        for feat, count in counts.items():
            i = feat_index.get(feat)
            if i is not None:
                idx.append(i)
                vals.append(count)
    else:
        for feat, i in feat_index.items():
            count = counts.get(feat)
            if count:
                idx.append(i)
                vals.append(count)
    return np.asarray(idx, dtype=np.intp), np.asarray(vals, dtype=COUNT_DTYPE)


def _zero_matrices(nf, nl, nd):
    """count_matrices' accumulators; shared with the workers so the dtypes
    cannot drift apart."""
    return (np.zeros((nf, nl), dtype=COUNT_DTYPE),
            np.zeros((nf, nd), dtype=COUNT_DTYPE),
            np.zeros((nf, nl), dtype=np.int8))


def _matrices_chunk(chunk):
    feat_index, lang_index, domain_index = shared()
    cm_lang, cm_domain, domcount = _zero_matrices(
        len(feat_index), len(lang_index), len(domain_index))
    for domain, lang, shard_path in chunk:
        idx, vals = _select_counts(load_shard(shard_path), feat_index)
        j = lang_index[lang]
        cm_lang[idx, j] += vals
        cm_domain[idx, domain_index[domain]] += vals
        # one shard = one (domain, lang) dir, so presence is a per-shard flag
        domcount[idx, j] += 1
    return cm_lang, cm_domain, domcount


def count_matrices(shard_items, features, lang_index, domain_index, jobs=None):
    """Everything the IG stage needs, in one pass over the shards.
    @returns (term x lang docfreq, term x domain docfreq, term x lang count
        of domains the term occurs in); the matrix rows follow `features`
    """
    feat_index = {f: i for i, f in enumerate(features)}
    cm_lang, cm_domain, domcount = _zero_matrices(
        len(features), len(lang_index), len(domain_index))
    with MapPool(jobs, set_shared, (feat_index, lang_index, domain_index)) as f:
        for part_lang, part_domain, part_dom in f(
                _matrices_chunk, job_chunks(shard_items, jobs)):
            cm_lang += part_lang
            cm_domain += part_domain
            domcount += part_dom
    return cm_lang, cm_domain, domcount
