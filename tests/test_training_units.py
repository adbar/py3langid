"""Unit tests for the numeric core of the training pipeline."""
import math
import os

import numpy as np
import pytest

from py3langid.train.common import (
    MAX_NGRAM_ORDER,
    TOKENIZE_ORDER,
    chunks,
    is_cjk_bigram,
    job_chunks,
)
from py3langid.train.shards import (
    COUNT_DTYPE,
    build_shards,
    count_matrices,
    doc_ngrams,
    load_shard,
    merge_docfreq,
)
from py3langid.train.stages import (
    compute_IG,
    compute_IG_binarized,
    entropy,
    ngram_select,
    select_LD_features,
)


@pytest.fixture
def tokenize_order2(monkeypatch):
    """Tokenize byte order 2 only, so a shard payload is small enough to
    assert on exactly. Moves the tokenizer and the cache key together, but
    NOT selection: SELECT_ORDERS is derived at import, so tests needing a
    different selection pass ngram_select's `orders`."""
    monkeypatch.setattr("py3langid.train.shards.MAX_NGRAM_ORDER", 2)


def test_entropy():
    assert entropy([1, 1]) == np.log(2)
    assert entropy([2, 0]) == 0.0
    assert entropy([1, 1, 1, 1]) == np.log(4)


def test_doc_ngrams():
    # orders below MIN_NGRAM_ORDER are not emitted: ngram_select filters on
    # length, so single bytes could never be selected as features
    assert doc_ngrams(b"abab", 2) == {b"ab", b"ba"}
    assert doc_ngrams(b"abab", 3) == {b"ab", b"ba", b"aba", b"bab"}
    assert doc_ngrams(b"", 2) == set()
    assert doc_ngrams(b"a", 2) == set()


def test_doc_ngrams_cjk_only_at_tokenize_order():
    """below TOKENIZE_ORDER, that order yields CJK codepoint bigrams only"""
    cjk = "\u4e2d\u6587".encode()          # two 3-byte CJK codepoints
    latin = b"abcdef"
    assert cjk in doc_ngrams(cjk, 5)
    assert len(cjk) == TOKENIZE_ORDER
    # no other 6-byte term survives
    assert {t for t in doc_ngrams(cjk + latin, 5) if len(t) == TOKENIZE_ORDER} == {cjk}
    assert not {t for t in doc_ngrams(latin, 5) if len(t) == TOKENIZE_ORDER}
    # asking for the full order restores every 6-gram
    assert latin in doc_ngrams(latin, TOKENIZE_ORDER)


def test_doc_ngrams_matches_unrestricted_selection():
    """the CJK restriction drops only unselectable terms"""
    data = "\u4e2d\u6587abc\u3042\u3044 xyz\u00e9\u00e8".encode()
    full = doc_ngrams(data, TOKENIZE_ORDER)
    selectable = {t for t in full if len(t) < TOKENIZE_ORDER or is_cjk_bigram(t)}
    assert doc_ngrams(data, TOKENIZE_ORDER - 1) == selectable


def test_ngram_select_restricts_tokenize_order_to_cjk():
    """order TOKENIZE_ORDER admits CJK bigrams only, whatever the shards
    hold there: at MAX_NGRAM_ORDER >= TOKENIZE_ORDER tokenization emits
    plain 6-grams, which must not become features"""
    cjk = "\u4e2d\u6587".encode()
    doc_count = {cjk: 5, b"abcdef": 99, b"ab": 3}
    selected = ngram_select(doc_count, 10, orders={2, TOKENIZE_ORDER})
    assert selected == [b"ab", cjk]  # b"abcdef" dropped despite the top DF


def test_compute_IG_nonbinarized():
    # 2 events with 2 docs each. b'aa' occurs only in event 0 (perfectly
    # discriminative, IG = log 2); b'bb' occurs once per event (IG = 0).
    cm = np.array([[2, 0], [1, 1]])
    dist = np.array([2, 2])
    ig = compute_IG(cm, dist)
    assert math.isclose(ig[0], math.log(2))
    assert math.isclose(ig[1], 0.0, abs_tol=1e-12)
    # degenerate terms score 0, not nan: absent everywhere, then present
    # everywhere (an all-zero count vector has no entropy to report)
    degenerate = compute_IG(np.array([[0, 0], [2, 2]]), dist)
    assert list(degenerate) == [0.0, 0.0]


def test_compute_IG_binarized():
    # Same data, binarized per event: shape (num_term, num_event),
    # same exact values by symmetry.
    cm = np.array([[2, 0], [1, 1]])
    dist = np.array([2, 2])
    ig = compute_IG_binarized(cm, dist)
    assert ig.shape == (2, 2)
    for event in range(2):
        assert math.isclose(ig[0, event], math.log(2))
        assert math.isclose(ig[1, event], 0.0, abs_tol=1e-12)
    assert not np.isnan(ig).any()
    # degenerate terms and a single-event corpus both score 0, not nan
    assert not compute_IG_binarized(np.array([[0, 0], [2, 2]]), dist).any()
    assert not compute_IG_binarized(np.array([[2], [0]]), np.array([2])).any()
    # chunking over terms cannot change a value
    big = np.array([[2, 0], [1, 1], [0, 2], [2, 2], [0, 0]])
    assert np.array_equal(compute_IG_binarized(big, dist, chunk=2),
                          compute_IG_binarized(big, dist, chunk=99))


def test_select_LD_features():
    # LD = IG_lang - IG_domain: term 2 is penalized for being domain-informative
    ld = np.array([
        [0.7, 0.1],
        [0.1, 0.6],
        [-0.1, -0.1],
    ])
    present = np.ones(ld.shape, dtype=bool)
    assert select_LD_features(ld, 1, present) == {0, 1}
    assert select_LD_features(ld, 3, present) == {0, 1, 2}
    # a language's picks are restricted to features present in it
    only_first = np.array([[True, True], [False, True], [False, True]])
    assert select_LD_features(ld, 3, only_first) == {0, 1, 2}
    assert select_LD_features(ld, 1, only_first) == {0, 1}


def test_ngram_select():
    doc_count = {b"a": 5, b"b": 3, b"ab": 10, b"cd": 1}
    feats = ngram_select(doc_count, tokens_per_order=1, orders={1, 2})
    assert feats == [b"a", b"ab"]


def test_build_shards_cache(tmp_path, tokenize_order2):
    lang_dir = tmp_path / "corpus" / "web" / "en"
    lang_dir.mkdir(parents=True)
    doc0 = lang_dir / "doc0.txt"
    doc0.write_bytes(b"abab")
    (lang_dir / "doc1.txt").write_bytes(b"ab")
    items = [("web", "en", str(lang_dir / f"doc{i}.txt")) for i in range(2)]
    shard_dir = str(tmp_path / "shards")

    [(domain, lang, shard_path)] = build_shards(items, shard_dir, jobs=1)
    assert (domain, lang) == ("web", "en")
    # shards carry document frequency only -- the NB numerators come from
    # feature_counts, so total occurrence counts are never stored
    docfreq = load_shard(shard_path)
    assert docfreq == {b"ab": 2, b"ba": 1}

    # unchanged corpus: shard is reused, not rewritten
    mtime = os.path.getmtime(shard_path)
    build_shards(items, shard_dir, jobs=1)
    assert os.path.getmtime(shard_path) == mtime

    # changed doc invalidates and rebuilds the shard
    doc0.write_bytes(b"zzzz")
    build_shards(items, shard_dir, jobs=1)
    docfreq = load_shard(shard_path)
    assert docfreq[b"ab"] == 1  # only doc1 still has it
    assert docfreq[b"zz"] == 1


def test_chunks_and_job_chunks():
    seq = list(range(10))
    assert chunks(seq, 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert chunks(seq, 0) == [[i] for i in seq]   # size floors at 1
    assert chunks([], 4) == []
    assert len(job_chunks(seq, 3)) == 3
    assert [x for c in job_chunks(seq, 3) for x in c] == seq
    assert job_chunks([], 3) == []


def test_merge_docfreq_spans_chunks(tmp_path, monkeypatch, tokenize_order2):
    """the merge reduces across several chunks, not just one"""
    monkeypatch.setattr("py3langid.train.shards.MERGE_SHARDS_PER_CHUNK", 2)
    items, shard_dir = [], str(tmp_path / "shards")
    for i in range(5):
        d = tmp_path / "corpus" / "web" / f"l{i}"
        d.mkdir(parents=True)
        (d / "doc0.txt").write_bytes(b"abab")
        items.append(("web", f"l{i}", str(d / "doc0.txt")))
    shard_items = build_shards(items, shard_dir, jobs=1)
    assert len(chunks(shard_items, 2)) == 3   # the path under test
    # every shard is b"abab": df 1 per shard, so 5 shards sum to 5
    assert merge_docfreq(shard_items, jobs=1) == {b"ab": 5, b"ba": 5}


def test_count_matrices(tmp_path, tokenize_order2):
    """per-lang/per-domain docfreq and domain presence, in one shard pass"""
    items, shard_dir = [], str(tmp_path / "shards")
    # en appears in two domains, fr in one
    for domain, lang, data in (("web", "en", b"abab"), ("news", "en", b"abab"),
                               ("web", "fr", b"cdcd")):
        d = tmp_path / "corpus" / domain / lang
        d.mkdir(parents=True)
        (d / "doc0.txt").write_bytes(data)
        items.append((domain, lang, str(d / "doc0.txt")))
    shard_items = build_shards(items, shard_dir, jobs=1)

    feats = [b"ab", b"cd"]
    lang_index, domain_index = {"en": 0, "fr": 1}, {"news": 0, "web": 1}
    cm_lang, cm_domain, domcount = count_matrices(
        shard_items, feats, lang_index, domain_index, jobs=1)

    assert cm_lang.dtype == COUNT_DTYPE and cm_domain.dtype == COUNT_DTYPE
    assert cm_lang.tolist() == [[2, 0], [0, 1]]    # b"ab" in 2 en docs
    assert cm_domain.tolist() == [[1, 1], [0, 1]]  # b"ab" in news + web
    assert domcount.tolist() == [[2, 0], [0, 1]]   # b"ab" in 2 en domains


def test_shard_cache_keyed_on_tokenization(tmp_path, monkeypatch):
    """Reuse is exact key equality: editing a tokenization constant rebuilds
    instead of serving shards written under the old one. (Reuse used to be
    order >=, which let features depend on the cache's history.)"""
    d = tmp_path / "corpus" / "web" / "zh"
    d.mkdir(parents=True)
    doc = d / "doc0.txt"
    doc.write_bytes("中文abcdef".encode())
    items = [("web", "zh", str(doc))]
    shard_dir = str(tmp_path / "shards")

    [(_, _, shard_path)] = build_shards(items, shard_dir, jobs=1)
    terms = load_shard(shard_path)
    # order TOKENIZE_ORDER is CJK-only however long the byte orders run
    assert {t for t in terms if len(t) == TOKENIZE_ORDER} == {"中文".encode()}
    assert max(len(t) for t in terms if len(t) != TOKENIZE_ORDER) == MAX_NGRAM_ORDER

    monkeypatch.setattr("py3langid.train.shards.MAX_NGRAM_ORDER", 3)
    build_shards(items, shard_dir, jobs=1)
    terms = load_shard(shard_path)
    # rebuilt at the new order: the 4- and 5-grams are gone, not inherited
    assert max(len(t) for t in terms if len(t) != TOKENIZE_ORDER) == 3
    assert {t for t in terms if len(t) == TOKENIZE_ORDER} == {"中文".encode()}


def test_select_counts_intersects_either_way():
    """both branches (iterate the shard vs the feature index) must agree"""
    from py3langid.train.shards import _select_counts

    feat_index = {b"ab": 0, b"cd": 1, b"ef": 2}

    def mapping(counts):
        idx, vals = _select_counts(counts, feat_index)
        assert idx.dtype == np.intp and vals.dtype == COUNT_DTYPE
        return dict(zip(idx.tolist(), vals.tolist()))

    assert mapping({b"ab": 3, b"zz": 9}) == {0: 3}                  # shard smaller
    assert mapping({b"ab": 3, b"zz": 9, b"cd": 1, b"qq": 2}) == {0: 3, 1: 1}
    assert mapping({}) == {}                                        # no overlap


def test_index_corpus_first_appearance_order(tmp_path):
    """class column order is first appearance along the sorted walk, NOT
    alphabetical -- it fixes nb_classes, so pin it"""
    from py3langid.train.stages import index_corpus

    for domain, langs in (("aaa", ["en", "fr"]), ("bbb", ["de", "en"])):
        for lang in langs:
            d = tmp_path / domain / lang
            d.mkdir(parents=True, exist_ok=True)
            (d / "doc0.txt").write_bytes(b"some text here")
    items, langs, domains = index_corpus(tmp_path)
    assert langs == ["en", "fr", "de"]   # "de" is absent from the first domain
    assert domains == ["aaa", "bbb"]
    assert len(items) == 4


def test_cluster_features():
    """a cluster spends its budget on non-junk features the quota missed,
    ranked by IG over the cluster's own languages"""
    from py3langid.train.stages import cluster_features

    feats = [b"11", b"aa", b"bb", b"cc"]
    # docfreq over langs (en, de, fr); IG within {en, de} descends 11 > aa > bb,
    # and cc is uninformative there
    cm_lang = np.array([[4, 0, 0], [3, 0, 0], [2, 0, 0], [2, 2, 0]])
    lang_dist = np.array([4, 4, 4])
    lang_index = {"en": 0, "de": 1, "fr": 2}

    def run(base, k, clusters=(("en", "de"),)):
        return cluster_features(cm_lang, lang_dist, lang_index, feats, base,
                                clusters, k)

    assert run(set(), 1) == {1}                # ranking: b"aa" beats b"bb"
    # b"11" is digits-only and b"aa" is already selected, so the best
    # *eligible* feature wins even though it ranks third by IG
    assert run({1}, 1) == {2}
    # no IG floor: once the eligible ranking is exhausted the budget takes
    # uninformative features too
    assert run({1}, 2) == {2, 3}
    assert run(set(), 3) == {1, 2, 3}          # only b"11" stays excluded
    assert run({1, 2, 3}, 2) == set()          # nothing eligible left
    # a cluster naming a language absent from the corpus is skipped entirely
    assert run(set(), 3, clusters=(("en", "xx"),)) == set()


def test_build_scanner_longest_match():
    """the DFA emits, at each byte position, the longest feature ending there"""
    from py3langid.train.scanner import build_scanner

    feats = [b"ab", b"abc", b"bc", b"c", b"xy", b"aab"]
    rows, row_index, out = build_scanner(feats)
    index = {f: i for i, f in enumerate(feats)}

    def walk(data):
        state, got = 0, []
        for byte in data:
            state = rows[(row_index[state] << 8) + byte]
            got.append(out[state])
        return got

    def longest_endings(data):
        res = []
        for end in range(1, len(data) + 1):
            hits = [f for f in feats if data[:end].endswith(f)]
            res.append(index[max(hits, key=len)] if hits else -1)
        return res

    for data in (b"", b"c", b"zzz", b"xy", b"xabcy", b"abcabc", b"aabc"):
        assert walk(data) == longest_endings(data), data


def test_build_scanner_shares_every_duplicate_row():
    """row sharing is maximal: no two stored rows hold the same transitions,
    which is what lets save_model canonicalize without deduplicating"""
    from py3langid.train.scanner import build_scanner

    feats = [b"ab", b"abc", b"bc", b"c", b"xy", b"aab", b"bca", b"cab"]
    rows, row_index, out = build_scanner(feats)
    stored = len(rows) // 256
    assert stored < len(out)  # sharing actually happened

    contents = {tuple(rows[(row_index[s] << 8):(row_index[s] << 8) + 256])
                for s in range(len(out))}
    assert len(contents) == stored


def test_feature_counts(tmp_path):
    """NB numerators = one longest match per byte position, per language"""
    from py3langid.train.scanner import build_scanner
    from py3langid.train.stages import feature_counts

    feats = [b"ab", b"abc", b"c"]
    rows, row_index, out = build_scanner(feats)

    docs = {"en": [b"abcabc", b"ab", b""], "fr": [b"cc", b"xabz"]}
    items = []
    for lang, texts in docs.items():
        for i, text in enumerate(texts):
            path = tmp_path / f"{lang}{i}.txt"
            path.write_bytes(text)
            items.append(("dom", lang, str(path)))
    lang_index = {"en": 0, "fr": 1}

    def brute(text):
        """count the longest feature ending at each position"""
        counts = [0] * len(feats)
        for end in range(1, len(text) + 1):
            hits = [f for f in feats if text[:end].endswith(f)]
            if hits:
                counts[feats.index(max(hits, key=len))] += 1
        return counts

    expected = np.zeros((len(feats), 2), dtype=np.int64)
    for lang, texts in docs.items():
        for text in texts:
            expected[:, lang_index[lang]] += brute(text)

    got = feature_counts(items, rows, row_index, out, len(feats), lang_index,
                         0, jobs=1)
    assert np.array_equal(got, expected)
    # worker partials are integer sums: the parallel result is exact
    assert np.array_equal(
        feature_counts(items, rows, row_index, out, len(feats), lang_index,
                       0, jobs=2), expected)
    # doc_cap truncates before counting
    capped = feature_counts(items, rows, row_index, out, len(feats),
                            lang_index, 2, jobs=1)
    assert capped.sum() < expected.sum()
