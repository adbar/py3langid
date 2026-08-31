"""
shards.py -
Per-(domain, lang) n-gram count shards.

One tokenization pass over the corpus produces, for each (domain, lang)
directory, a term -> document frequency dict covering every selectable term
(see doc_ngrams). Feature selection is algebra over these shards, so corpus
edits only retokenize the directories they touch and setting changes
retokenize nothing. (The NB parameters are counted separately, by
feature_counts, because the runtime credits one longest match per byte
position rather than every n-gram occurrence.)

Selection needs exactly two passes: merge_docfreq tallies globally to
choose the candidate pool, then count_matrices projects the shards onto
it. Everything the IG stage needs -- per-lang and per-domain document
frequency plus per-lang domain presence -- comes out of that second pass.

Shard file layout: two marshal objects, a (key, max_order) header
followed by the docfreq payload. The key hashes the shard directory's
(filename, size, mtime) list, so it is location-independent. A cached shard
is reused when its max_order is at least the one asked for; shards written
before the CJK-only rule record TOKENIZE_ORDER and are still supersets.

All aggregations parallelize over shards and reduce integer partial sums,
which keeps them exact and deterministic regardless of worker scheduling.
"""

import hashlib
import marshal
import os
from collections import Counter, defaultdict

import numpy as np

from .common import (
    MIN_NGRAM_ORDER,
    TOKENIZE_ORDER,
    MapPool,
    is_cjk_bigram,
    job_count,
    read_doc,
)

# small chunks keep each worker's partial Counter small (one chunk per job
# grew a near-global Counter in every worker: 8.7 -> 4.7 GB peak)
MERGE_SHARDS_PER_CHUNK = 8

# document frequencies, bounded by docs per class (~10^3); int32 halves the
# count matrices, which every worker allocates in full and ships back
COUNT_DTYPE = np.int32


def doc_ngrams(data, max_order):
    """Distinct terms one doc contributes: byte n-grams of order
    MIN_NGRAM_ORDER..max_order, plus -- when max_order stops short of it --
    only the CJK codepoint bigrams at TOKENIZE_ORDER, the rest of that order
    being unselectable. Orders below MIN_NGRAM_ORDER are unselectable too."""
    terms = set()
    n = len(data)
    cjk_order = TOKENIZE_ORDER if max_order < TOKENIZE_ORDER else 0
    for i in range(n):
        for k in range(MIN_NGRAM_ORDER, min(max_order, n - i) + 1):
            terms.add(data[i:i + k])
        # only 3+3 bytes can pass is_cjk_bigram, and a codepoint >= U+2E80
        # has lead byte >= 0xE2: two comparisons skip almost every position
        if cjk_order and i + cjk_order <= n \
                and data[i] >= 0xE2 and data[i + 3] >= 0xE2:
            term = data[i:i + cjk_order]
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
    h = hashlib.sha256(f"cap{doc_cap}\0".encode())
    for p in sorted(paths):
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return h.hexdigest()


def _chunks(seq, size):
    """Split seq into contiguous chunks of at most `size` items."""
    size = max(1, size)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _job_chunks(seq, jobs):
    """One contiguous chunk per job: for reducers whose partial is a dense
    matrix, where more chunks would mean more transfers."""
    return _chunks(seq, -(-len(seq) // job_count(jobs)))


def _setup_build(max_order, doc_cap):
    global __max_order, __doc_cap
    __max_order = max_order
    __doc_cap = doc_cap


def _build_shard(arg):
    """Build one shard unless a valid cached one exists.
    @returns (shard_path, built)
    """
    shard_path, key, paths = arg
    try:
        with open(shard_path, 'rb') as f:
            old_key, old_order = marshal.load(f)
        if old_key == key and old_order >= __max_order:
            return shard_path, False
    except (OSError, EOFError, ValueError, TypeError):
        pass

    docfreq = Counter()
    for path in paths:
        docfreq.update(doc_ngrams(read_doc(path, __doc_cap), __max_order))

    tmp_path = shard_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        marshal.dump((key, __max_order), f)
        marshal.dump(dict(docfreq), f)
    os.replace(tmp_path, shard_path)
    return shard_path, True


def build_shards(items, shard_dir, max_order, jobs=None, doc_cap=0):
    """Build (or reuse cached) shards for all (domain, lang) groups.

    @param items (domain, lang, path) triples with string names
    @param doc_cap truncate each doc to this many bytes (0 = no cap);
        part of the shard filename and cache key, so one shard dir
        serves any mix of caps
    @returns list of (domain, lang, shard_path), sorted by (domain, lang)
    """
    os.makedirs(shard_dir, exist_ok=True)
    tasks = []
    shard_items = []
    for (domain, lang), paths in group_items(items):
        shard_path = os.path.join(shard_dir, f"{domain}__{lang}.cap{doc_cap}")
        tasks.append((shard_path, _group_key(paths, doc_cap), paths))
        shard_items.append((domain, lang, shard_path))

    with MapPool(jobs, _setup_build, (max_order, doc_cap)) as f:
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
                         _chunks(shard_items, MERGE_SHARDS_PER_CHUNK)):
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


def _setup_counts(feat_index, lang_index, domain_index):
    global __feat_index, __lang_index, __domain_index
    __feat_index = feat_index
    __lang_index = lang_index
    __domain_index = domain_index


def _zero_matrices(nf, nl, nd):
    """count_matrices' accumulators; shared with the workers so the dtypes
    cannot drift apart."""
    return (np.zeros((nf, nl), dtype=COUNT_DTYPE),
            np.zeros((nf, nd), dtype=COUNT_DTYPE),
            np.zeros((nf, nl), dtype=np.int8))


def _matrices_chunk(chunk):
    cm_lang, cm_domain, domcount = _zero_matrices(
        len(__feat_index), len(__lang_index), len(__domain_index))
    for domain, lang, shard_path in chunk:
        idx, vals = _select_counts(load_shard(shard_path), __feat_index)
        j = __lang_index[lang]
        cm_lang[idx, j] += vals
        cm_domain[idx, __domain_index[domain]] += vals
        # one shard = one (domain, lang) dir, so presence is a per-shard flag
        domcount[idx, j] += 1
    return cm_lang, cm_domain, domcount


def count_matrices(shard_items, features, lang_index, domain_index, jobs=None):
    """Everything the IG stage needs, in one pass over the shards.
    @returns (term x lang docfreq, term x domain docfreq, term x lang count
        of domains the term occurs in, {lang: num_domains}); the matrix rows
        follow `features`
    """
    feat_index = {f: i for i, f in enumerate(features)}
    cm_lang, cm_domain, domcount = _zero_matrices(
        len(features), len(lang_index), len(domain_index))
    with MapPool(jobs, _setup_counts, (feat_index, lang_index, domain_index)) as f:
        for part_lang, part_domain, part_dom in f(
                _matrices_chunk, _job_chunks(shard_items, jobs)):
            cm_lang += part_lang
            cm_domain += part_domain
            domcount += part_dom
    # one shard per (domain, lang) dir, so this is just a tally of shards
    lang_domains = Counter(lang for _, lang, _ in shard_items)
    return cm_lang, cm_domain, domcount, lang_domains
