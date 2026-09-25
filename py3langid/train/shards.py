"""Per-(domain, lang) tokenization shards, cached by content: n-gram document
frequencies plus the word counts and CJK character frequencies of the token table."""

import hashlib
import marshal
import os
from collections import Counter
from itertools import groupby
from operator import itemgetter

import numpy as np

from ..langid import CJK_RE, TOKEN_RE
from .common import (
    DOC_CAP,
    MAX_NGRAM_ORDER,
    MIN_NGRAM_ORDER,
    NORMALIZE_VERSION,
    MapPool,
    pmap_chunks,
    read_doc,
)

COUNT_DTYPE = np.int32


def doc_ngrams(data, max_order):
    """Distinct byte n-grams in a doc."""
    terms = set()
    n = len(data)
    for i in range(n):
        for k in range(MIN_NGRAM_ORDER, min(max_order, n - i) + 1):
            terms.add(data[i:i + k])
    return terms


def doc_tokens(text):
    """(words, distinct CJK characters) of a doc."""
    tokens = TOKEN_RE.findall(text)
    cjk = {t for t in tokens if len(t) == 1 and CJK_RE.match(t)}
    return [t for t in tokens if t not in cjk], cjk


def group_items(items):
    """Group by (domain, lang), sorted."""
    return [(k, [p for *_, p in g]) for k, g in groupby(sorted(items), itemgetter(0, 1))]


def _group_key(paths):
    """Cache key from doc metadata + tokenization constants."""
    h = hashlib.sha256()
    h.update(f"{MIN_NGRAM_ORDER}\0{MAX_NGRAM_ORDER}\0"
             f"{DOC_CAP}\0{NORMALIZE_VERSION}\0".encode())
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

    docfreq, words, chars = Counter(), Counter(), Counter()
    for path in paths:
        data, text = read_doc(path, DOC_CAP)
        docfreq.update(doc_ngrams(data, MAX_NGRAM_ORDER))
        w, c = doc_tokens(text)
        words.update(w)
        chars.update(c)

    tmp_path = shard_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        marshal.dump(key, f)
        marshal.dump((dict(words), dict(chars), len(paths)), f)  # small part first: load_tokens skips docfreq
        marshal.dump(dict(docfreq), f)
    os.replace(tmp_path, shard_path)
    return shard_path, True


def build_shards(items, shard_dir, jobs=1):
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


def load_tokens(shard_path):
    """(word counts, CJK char document frequency, doc count) of a shard."""
    with open(shard_path, 'rb') as f:
        marshal.load(f)
        return marshal.load(f)


def load_shard(shard_path):
    """Load term → document frequency dict."""
    with open(shard_path, 'rb') as f:
        marshal.load(f)
        marshal.load(f)
        return marshal.load(f)


def _merge_chunk(chunk):
    merged = Counter()
    for _, _, shard_path in chunk:
        merged.update(load_shard(shard_path))
    return merged


def merge_docfreq(shard_items, jobs=1):
    """Global term → document frequency."""
    doc_count = Counter()
    for partial in pmap_chunks(_merge_chunk, shard_items, jobs):
        doc_count.update(partial)
    return doc_count


def _select_counts(counts, feat_index):
    """Intersect shard counts with feature index. Returns (indices, counts)."""
    idx, vals = [], []
    for feat, count in counts.items():
        i = feat_index.get(feat)
        if i is not None:
            idx.append(i)
            vals.append(count)
    return np.asarray(idx, dtype=np.intp), np.asarray(vals, dtype=COUNT_DTYPE)


def _matrices_chunk(feat_index, lang_index, domain_index, chunk):
    cm_lang = np.zeros((len(feat_index), len(lang_index)), dtype=COUNT_DTYPE)
    cm_domain = np.zeros((len(feat_index), len(domain_index)), dtype=COUNT_DTYPE)
    for domain, lang, shard_path in chunk:
        idx, vals = _select_counts(load_shard(shard_path), feat_index)
        cm_lang[idx, lang_index[lang]] += vals
        cm_domain[idx, domain_index[domain]] += vals
    return cm_lang, cm_domain


def count_matrices(shard_items, features, lang_index, domain_index, jobs=1):
    """Returns (lang counts, domain counts) document-frequency matrices."""
    feat_index = {f: i for i, f in enumerate(features)}
    cm_lang = np.zeros((len(features), len(lang_index)), dtype=COUNT_DTYPE)
    cm_domain = np.zeros((len(features), len(domain_index)), dtype=COUNT_DTYPE)
    for part_lang, part_domain in pmap_chunks(
            _matrices_chunk, shard_items, jobs, (feat_index, lang_index, domain_index)):
        cm_lang += part_lang
        cm_domain += part_domain
    return cm_lang, cm_domain
