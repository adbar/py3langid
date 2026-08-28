"""
shards.py -
Per-(domain, lang) n-gram count shards.

One tokenization pass over the corpus produces, for each (domain, lang)
directory, two dicts covering all byte n-grams of order 1..max_order:
term -> document frequency and term -> total occurrence count. Every
downstream training stage is algebra over these shards, so corpus edits
only retokenize the directories they touch and setting changes retokenize
nothing.

Shard file layout: two marshal objects, a (key, max_order) header
followed by the (docfreq, totalfreq) payload. The key hashes the shard
directory's (filename, size, mtime) list, so it is location-independent.

All aggregations parallelize over shards and reduce integer partial sums,
which keeps them exact and deterministic regardless of worker scheduling.
"""

import hashlib
import marshal
import os
from collections import Counter, defaultdict

import numpy as np

from .common import MapPool, job_count, read_doc


def count_ngrams(data, max_order):
    """term -> occurrence count over all byte n-grams of order 1..max_order."""
    counts = Counter()
    n = len(data)
    for i in range(n):
        for k in range(1, min(max_order, n - i) + 1):
            counts[data[i:i + k]] += 1
    return counts


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


def _chunk(seq, jobs):
    """Split seq into one contiguous chunk per job."""
    pieces = job_count(jobs)
    size = max(1, -(-len(seq) // pieces))
    return [seq[i:i + size] for i in range(0, len(seq), size)]


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
    totalfreq = Counter()
    for path in paths:
        counts = count_ngrams(read_doc(path, __doc_cap), __max_order)
        totalfreq.update(counts)
        docfreq.update(counts.keys())

    tmp_path = shard_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        marshal.dump((key, __max_order), f)
        marshal.dump((dict(docfreq), dict(totalfreq)), f)
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
    """@returns the (docfreq, totalfreq) dicts of a shard."""
    with open(shard_path, 'rb') as f:
        marshal.load(f)
        return marshal.load(f)


def _merge_chunk(chunk):
    merged = Counter()
    for _, _, shard_path in chunk:
        docfreq, _ = load_shard(shard_path)
        merged.update(docfreq)
    return merged


def merge_docfreq(shard_items, jobs=None):
    """Global term -> document frequency over all shards."""
    doc_count = Counter()
    with MapPool(jobs) as f:
        for partial in f(_merge_chunk, _chunk(shard_items, jobs)):
            doc_count.update(partial)
    return doc_count


def _select_counts(counts, feat_index):
    """Intersect a shard count dict with a feature index, iterating the
    smaller of the two.
    @returns (row indices, counts) with unique indices
    """
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
    return idx, vals


def _setup_counts(feat_index, lang_index, domain_index):
    global __feat_index, __lang_index, __domain_index
    __feat_index = feat_index
    __lang_index = lang_index
    __domain_index = domain_index


def _matrices_chunk(chunk):
    cm_lang = np.zeros((len(__feat_index), len(__lang_index)), dtype=int)
    cm_domain = np.zeros((len(__feat_index), len(__domain_index)), dtype=int)
    prod = np.zeros((len(__feat_index), len(__lang_index)), dtype=int)
    for domain, lang, shard_path in chunk:
        docfreq, totalfreq = load_shard(shard_path)
        idx, vals = _select_counts(docfreq, __feat_index)
        cm_lang[idx, __lang_index[lang]] += vals
        cm_domain[idx, __domain_index[domain]] += vals
        idx, vals = _select_counts(totalfreq, __feat_index)
        prod[idx, __lang_index[lang]] += vals
    return cm_lang, cm_domain, prod


def domain_presence(shard_items, features, lang_index):
    """@returns ((num_term, num_lang) count of domains in which the term
    occurs for that language, {lang: num_domains})"""
    pos = {f: i for i, f in enumerate(features)}
    counts = np.zeros((len(features), len(lang_index)), dtype=np.int8)
    feat_set = set(features)
    lang_domains = Counter()
    for _domain, lang, shard_path in shard_items:
        docfreq, _ = load_shard(shard_path)
        j = lang_index[lang]
        lang_domains[lang] += 1
        for f in feat_set.intersection(docfreq):
            counts[pos[f], j] += 1
    return counts, lang_domains


def count_matrices(shard_items, features, lang_index, domain_index, jobs=None):
    """Document-frequency count matrices for the IG computation, plus the
    total-occurrence matrix (the NB numerators), in one pass over the shards.
    @returns (term x lang docfreq, term x domain docfreq, term x lang
        totalfreq) int arrays, rows follow `features`
    """
    feat_index = {f: i for i, f in enumerate(features)}
    cm_lang = np.zeros((len(features), len(lang_index)), dtype=int)
    cm_domain = np.zeros((len(features), len(domain_index)), dtype=int)
    prod = np.zeros((len(features), len(lang_index)), dtype=int)
    with MapPool(jobs, _setup_counts, (feat_index, lang_index, domain_index)) as f:
        for part_lang, part_domain, part_prod in f(_matrices_chunk, _chunk(shard_items, jobs)):
            cm_lang += part_lang
            cm_domain += part_domain
            prod += part_prod
    return cm_lang, cm_domain, prod
