"""Per-(domain, lang) n-gram document-frequency shards with content-based caching."""

import hashlib
import marshal
import os
from collections import Counter, defaultdict

import numpy as np

from .common import (
    DOC_CAP,
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

MERGE_SHARDS_PER_CHUNK = 8
COUNT_DTYPE = np.int32


def doc_ngrams(data, max_order):
    """Distinct byte n-grams in a doc, plus CJK codepoint bigrams."""
    terms = set()
    n = len(data)
    for i in range(n):
        for k in range(MIN_NGRAM_ORDER, min(max_order, n - i) + 1):
            terms.add(data[i:i + k])
        if i + TOKENIZE_ORDER <= n \
                and data[i] >= 0xE2 and data[i + 3] >= 0xE2:
            term = data[i:i + TOKENIZE_ORDER]
            if is_cjk_bigram(term):
                terms.add(term)
    return terms


def group_items(items):
    """Group by (domain, lang), sorted."""
    groups = defaultdict(list)
    for domain, lang, path in items:
        groups[(domain, lang)].append(path)
    return sorted(groups.items())


def _group_key(paths):
    """Cache key from doc metadata + tokenization constants."""
    h = hashlib.sha256()
    h.update(f"{MIN_NGRAM_ORDER}\0{MAX_NGRAM_ORDER}\0{TOKENIZE_ORDER}\0"
             f"{DOC_CAP}\0".encode())
    for p in sorted(paths):
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return h.hexdigest()


def _build_shard(arg):
    """Build one shard or reuse cached. Returns (shard_path, built)."""
    shard_path, key, paths = arg
    try:
        with open(shard_path, 'rb') as f:
            if marshal.load(f) == key:
                return shard_path, False
    except (OSError, EOFError, ValueError, TypeError):
        pass

    docfreq = Counter()
    for path in paths:
        docfreq.update(doc_ngrams(read_doc(path, DOC_CAP), MAX_NGRAM_ORDER))

    tmp_path = shard_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        marshal.dump(key, f)
        marshal.dump(dict(docfreq), f)
    os.replace(tmp_path, shard_path)
    return shard_path, True


def build_shards(items, shard_dir, jobs=None):
    """Build/reuse cached shards. Returns [(domain, lang, shard_path), ...]."""
    os.makedirs(shard_dir, exist_ok=True)
    tasks = []
    shard_items = []
    for (domain, lang), paths in group_items(items):
        shard_path = os.path.join(shard_dir, f"{domain}__{lang}")
        tasks.append((shard_path, _group_key(paths), paths))
        shard_items.append((domain, lang, shard_path))

    with MapPool(jobs) as f:
        built = sum(new for _, new in f(_build_shard, tasks))
    print(f"shards: {built} built, {len(tasks) - built} cached")
    return shard_items


def load_shard(shard_path):
    """Load term → document frequency dict."""
    with open(shard_path, 'rb') as f:
        marshal.load(f)
        return marshal.load(f)


def _merge_chunk(chunk):
    merged = Counter()
    for _, _, shard_path in chunk:
        merged.update(load_shard(shard_path))
    return merged


def merge_docfreq(shard_items, jobs=None):
    """Global term → document frequency."""
    doc_count = Counter()
    with MapPool(jobs) as f:
        for partial in f(_merge_chunk,
                         chunks(shard_items, MERGE_SHARDS_PER_CHUNK)):
            doc_count.update(partial)
    return doc_count


def _select_counts(counts, feat_index):
    """Intersect shard counts with feature index. Returns (indices, counts)."""
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
        domcount[idx, j] += 1
    return cm_lang, cm_domain, domcount


def count_matrices(shard_items, features, lang_index, domain_index, jobs=None):
    """Returns (lang counts, domain counts, domain-presence) matrices."""
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
