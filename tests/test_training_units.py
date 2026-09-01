"""Unit tests for the numeric core of the training pipeline."""
import math
import os
from pathlib import Path

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
    entropy,
    ld_weights,
    ngram_select,
    select_LD_features,
)


def make_corpus(tmp_path, docs):
    """Write tmp_path/corpus/<domain>/<lang>/docN.txt from (domain, lang, data)
    triples. @returns (build_shards items, shard dir)"""
    items = []
    for domain, lang, data in docs:
        lang_dir = tmp_path / "corpus" / domain / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        seen = sum(1 for item in items if item[:2] == (domain, lang))
        path = lang_dir / f"doc{seen}.txt"
        path.write_bytes(data)
        items.append((domain, lang, str(path)))
    return items, str(tmp_path / "shards")


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


def _ld_matrix(cm, dist, domain_ig=None):
    """ld_weights' columns stacked, for tests that want the whole matrix."""
    if domain_ig is None:
        domain_ig = np.zeros(len(cm))
    return np.stack(list(ld_weights(cm, dist, domain_ig)), axis=1)


def test_ld_weights():
    # Binarized per language: same data and, by symmetry, the same exact
    # values as the non-binarized IG above.
    cm = np.array([[2, 0], [1, 1]])
    dist = np.array([2, 2])
    ig = _ld_matrix(cm, dist)
    assert ig.shape == (2, 2)
    for lang in range(2):
        assert math.isclose(ig[0, lang], math.log(2))
        assert math.isclose(ig[1, lang], 0.0, abs_tol=1e-12)
    assert not np.isnan(ig).any()
    # degenerate terms and a single-language corpus both score 0, not nan
    assert not _ld_matrix(np.array([[0, 0], [2, 2]]), dist).any()
    assert not _ld_matrix(np.array([[2], [0]]), np.array([2])).any()
    # one column per language, in column order, whatever the term count
    big = np.array([[2, 0], [1, 1], [0, 2], [2, 2], [0, 0]])
    assert len(list(ld_weights(big, dist, np.zeros(5)))) == 2
    # the domain IG is a per-term offset subtracted from every column
    domain_ig = np.array([0.25, 0.5])
    assert np.array_equal(_ld_matrix(cm, dist, domain_ig),
                          _ld_matrix(cm, dist) - domain_ig[:, None])



def test_ld_weights_matches_contingency_table():
    """the fused per-column formula against a brute-force 2x2 IG reference"""
    def brute(cm, dist):
        cm, dist = np.asarray(cm, float), np.asarray(dist, float)
        n = dist.sum()

        def H(*ps):
            return -sum(p * math.log(p) for p in ps if p > 0)

        out = np.zeros(cm.shape)
        for i in range(cm.shape[0]):
            t = cm[i].sum()
            for j in range(cm.shape[1]):
                a, b, c = cm[i, j], t - cm[i, j], dist[j] - cm[i, j]
                d = (n - t) - c
                cond = (t / n) * H(a / t, b / t) if t else 0.0
                if n - t:
                    cond += ((n - t) / n) * H(c / (n - t), d / (n - t))
                out[i, j] = H(dist[j] / n, (n - dist[j]) / n) - cond
        return out

    rng = np.random.default_rng(0)
    cases = []
    for _ in range(40):
        nl, nt = int(rng.integers(1, 7)), int(rng.integers(1, 12))
        dist = rng.integers(1, 40, nl)
        cases.append((np.array([[rng.integers(0, dist[j] + 1) for j in range(nl)]
                                for _ in range(nt)], dtype=np.int32), dist))
    # all-zero rows, a single language, an entirely empty matrix
    cases += [(np.array([[0, 0], [2, 2]], dtype=np.int32), np.array([2, 2])),
              (np.array([[2], [0]], dtype=np.int32), np.array([2])),
              (np.zeros((3, 4), dtype=np.int32), np.array([5, 5, 5, 5]))]
    for cm, dist in cases:
        got = _ld_matrix(cm, dist)
        assert not np.isnan(got).any()
        assert np.allclose(got, brute(cm, dist), atol=1e-12)

def test_select_LD_features():
    # LD = IG_lang - IG_domain: term 2 is penalized for being domain-informative
    ld = np.array([
        [0.7, 0.1],
        [0.1, 0.6],
        [-0.1, -0.1],
    ])
    present = np.ones(ld.shape, dtype=bool)
    assert select_LD_features(ld.T, 1, present) == {0, 1}
    assert select_LD_features(ld.T, 3, present) == {0, 1, 2}
    # a language's picks are restricted to features present in it
    only_first = np.array([[True, True], [False, True], [False, True]])
    assert select_LD_features(ld.T, 3, only_first) == {0, 1, 2}
    assert select_LD_features(ld.T, 1, only_first) == {0, 1}


def test_ngram_select():
    doc_count = {b"a": 5, b"b": 3, b"ab": 10, b"cd": 1}
    feats = ngram_select(doc_count, tokens_per_order=1, orders={1, 2})
    assert feats == [b"a", b"ab"]


def test_build_shards_cache(tmp_path, tokenize_order2):
    items, shard_dir = make_corpus(tmp_path, [("web", "en", b"abab"),
                                              ("web", "en", b"ab")])
    doc0 = Path(items[0][2])

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
    items, shard_dir = make_corpus(
        tmp_path, [("web", f"l{i}", b"abab") for i in range(5)])
    shard_items = build_shards(items, shard_dir, jobs=1)
    assert len(chunks(shard_items, 2)) == 3   # the path under test
    # every shard is b"abab": df 1 per shard, so 5 shards sum to 5
    assert merge_docfreq(shard_items, jobs=1) == {b"ab": 5, b"ba": 5}


def test_count_matrices(tmp_path, tokenize_order2):
    """per-lang/per-domain docfreq and domain presence, in one shard pass"""
    # en appears in two domains, fr in one
    items, shard_dir = make_corpus(tmp_path, [("web", "en", b"abab"),
                                              ("news", "en", b"abab"),
                                              ("web", "fr", b"cdcd")])
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
    items, shard_dir = make_corpus(
        tmp_path, [("web", "zh", "中文abcdef".encode())])

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


def longest_endings(feats, data):
    """Reference scanner, naive O(n*|feats|): index of the longest feature
    ending at each byte position, -1 where none does."""
    res = []
    for end in range(1, len(data) + 1):
        hits = [f for f in feats if data[:end].endswith(f)]
        res.append(feats.index(max(hits, key=len)) if hits else -1)
    return res


@pytest.fixture(scope="module")
def longest_match_dfa():
    from py3langid.train.scanner import build_scanner

    feats = [b"ab", b"abc", b"bc", b"c", b"xy", b"aab"]
    return feats, build_scanner(feats)


@pytest.mark.parametrize("data", [b"", b"c", b"zzz", b"xy", b"xabcy",
                                 b"abcabc", b"aabc"])
def test_build_scanner_longest_match(longest_match_dfa, data):
    """the DFA emits, at each byte position, the longest feature ending there"""
    feats, (rows, row_index, out) = longest_match_dfa
    state, got = 0, []
    for byte in data:
        state = rows[(row_index[state] << 8) + byte]
        got.append(out[state])
    assert got == longest_endings(feats, data)


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


def test_feature_counts(tmp_path, monkeypatch):
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

    expected = np.zeros((len(feats), 2), dtype=np.int64)
    for lang, texts in docs.items():
        for text in texts:
            for feat in longest_endings(feats, text):
                if feat >= 0:
                    expected[feat, lang_index[lang]] += 1

    got = feature_counts(items, rows, row_index, out, len(feats), lang_index,
                         jobs=1)
    assert np.array_equal(got, expected)
    # worker partials are integer sums: the parallel result is exact
    assert np.array_equal(
        feature_counts(items, rows, row_index, out, len(feats), lang_index,
                       jobs=2), expected)
    # DOC_CAP truncates before counting
    monkeypatch.setattr("py3langid.train.stages.DOC_CAP", 2)
    capped = feature_counts(items, rows, row_index, out, len(feats),
                            lang_index, jobs=1)
    assert capped.sum() < expected.sum()
