"""Unit tests for the numeric core of the training pipeline."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from py3langid.langid import normalize
from py3langid.train import words as W
from py3langid.train.common import MAX_NGRAM_ORDER, read_doc, walk_corpus
from py3langid.train.scanner import build_scanner
from py3langid.train.shards import (
    COUNT_DTYPE,
    TOKENS,
    _build_shard,
    _group_key,
    build_shards,
    count_matrices,
    doc_ngrams,
    load_shard,
    merge_docfreq,
)
from py3langid.train.stages import (
    compute_IG,
    entropy,
    feature_counts,
    ld_weights,
    ngram_select,
    select_LD_features,
)
from py3langid.train.train import _axis

from .conftest import write_corpus


def make_corpus(tmp_path, docs):
    """(build_shards items, shard dir) of a corpus under tmp_path/corpus."""
    return write_corpus(tmp_path / "corpus", docs), str(tmp_path / "shards")


def test_entropy():
    assert entropy([1, 1]) == np.log(2)
    assert entropy([2, 0]) == 0.0
    assert entropy([1, 1, 1, 1]) == np.log(4)


def test_doc_ngrams():
    # orders below MIN_NGRAM_ORDER are never emitted, so a single byte
    # cannot become a feature
    assert doc_ngrams(b"abab", 2) == {b"ab", b"ba"}
    assert doc_ngrams(b"abab", 3) == {b"ab", b"ba", b"aba", b"bab"}
    assert doc_ngrams(b"", 2) == set()
    assert doc_ngrams(b"a", 2) == set()


def test_doc_ngrams_bounded_by_max_order():
    """every term sits between MIN_NGRAM_ORDER and the requested order, CJK included"""
    data = "中文abcあい xyzéè".encode()
    terms = doc_ngrams(data, MAX_NGRAM_ORDER)
    assert {len(t) for t in terms} == set(range(2, MAX_NGRAM_ORDER + 1))
    # a CJK character is 3 bytes, so a pair only ever appears as a 5-byte prefix
    assert "中文".encode() not in terms
    assert "中文".encode()[:5] in terms


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


def test_ld_weights_matches_contingency_table():
    """the fused per-column formula against a brute-force 2x2 IG reference, minus the domain IG"""
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
        domain_ig = np.arange(len(cm)) / 4
        assert np.array_equal(_ld_matrix(cm, dist, domain_ig), got - domain_ig[:, None])


def test_select_LD_features():
    # terms 0/1 are exclusive to one language each, term 2 is shared and
    # penalized as domain-informative (LD = IG_lang - IG_domain), term 3 absent
    cm = np.array([[3, 0], [0, 3], [1, 1], [0, 0]])
    dist = np.array([3, 3])
    domain_ig = np.array([0.0, 0.0, 0.5, 0.0])
    assert select_LD_features(cm, dist, domain_ig, 1) == {0, 1}
    assert select_LD_features(cm, dist, domain_ig, 2) == {0, 1, 2}
    # a language's picks are restricted to features present in it
    assert select_LD_features(cm, dist, domain_ig, 10) == {0, 1, 2}
    # only term 2 is present in both: for lang 0 it outranks term 1 (absent)
    ld = _ld_matrix(cm, dist, domain_ig)
    assert ld[1, 0] > ld[2, 0]
    assert select_LD_features(cm, dist, np.zeros(4), 1) == {0, 1}
    with pytest.raises(ValueError):
        select_LD_features(cm, dist, domain_ig, 0)


def test_ngram_select_per_order_pool():
    """tokens_per_order is a per-length budget, ties break on the term"""
    doc_count = {b"a": 9, b"b": 8, b"ab": 7, b"abc": 6, b"abd": 6}
    assert ngram_select(doc_count, 10) == [b"a", b"ab", b"abc", b"abd", b"b"]
    assert ngram_select(doc_count, 1) == [b"a", b"ab", b"abc"]
    assert ngram_select({}, 10) == []


@pytest.mark.parametrize("doc_cap,expected", [
    (3, [b"ab", b"bc"]),  # cap 3 sees only b"abc": b"cd" on is never tokenized
    (0, [b"ab", b"bc", b"cd", b"de", b"ef"]),  # 0 = no cap
])
def test_doc_cap_truncates(tmp_path, monkeypatch, tokenize_order2, doc_cap,
                           expected):
    """DOC_CAP bounds how much of a document reaches the tokenizer"""
    monkeypatch.setattr("py3langid.train.shards.DOC_CAP", doc_cap)
    doc = tmp_path / "doc0000.txt"
    doc.write_bytes(b"abcdef")
    shard = tmp_path / "shard"
    _build_shard((str(shard), _group_key([str(doc)]), [str(doc)]))
    # the bigrams of b"abcdef" are distinct: one doc each
    assert load_shard(str(shard)) == dict.fromkeys(expected, 1)


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
    assert load_shard(shard_path, TOKENS) == ({"abab": 1, "ab": 1}, {}, 2)  # word df, CJK df, docs

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


def test_merge_docfreq_spans_chunks(tmp_path, tokenize_order2):
    """the merge reduces across several chunks, not just one"""
    items, shard_dir = make_corpus(
        tmp_path, [("web", f"l{i}", b"abab") for i in range(5)])
    shard_items = build_shards(items, shard_dir, jobs=1)  # 5 shards over 3 jobs: 3 chunks
    # every shard is b"abab": df 1 per shard, so 5 shards sum to 5
    assert merge_docfreq(shard_items, jobs=3) == {b"ab": 5, b"ba": 5}


def test_count_matrices(tmp_path, tokenize_order2):
    """per-lang and per-domain docfreq in one shard pass"""
    # en appears in two domains, fr in one
    items, shard_dir = make_corpus(tmp_path, [("web", "en", b"abab"),
                                              ("news", "en", b"abab"),
                                              ("web", "fr", b"cdcd")])
    shard_items = build_shards(items, shard_dir, jobs=1)

    feats = [b"ab", b"cd"]
    lang_index, domain_index = {"en": 0, "fr": 1}, {"news": 0, "web": 1}
    cm_lang, cm_domain = count_matrices(
        shard_items, feats, lang_index, domain_index, jobs=1)

    assert cm_lang.dtype == COUNT_DTYPE and cm_domain.dtype == COUNT_DTYPE
    assert cm_lang.tolist() == [[2, 0], [0, 1]]    # b"ab" in 2 en docs
    assert cm_domain.tolist() == [[1, 1], [0, 1]]  # b"ab" in news + web


def test_shard_cache_keyed_on_tokenization(tmp_path, monkeypatch):
    """Reuse is exact key equality: editing a tokenization constant (order, cap) rebuilds
    instead of serving shards written under the old one. (Reuse used to be
    order >=, which let features depend on the cache's history.)"""
    items, shard_dir = make_corpus(
        tmp_path, [("web", "zh", "中文abcdef".encode())])

    [(_, _, shard_path)] = build_shards(items, shard_dir, jobs=1)
    terms = load_shard(shard_path)
    assert max(len(t) for t in terms) == MAX_NGRAM_ORDER

    monkeypatch.setattr("py3langid.train.shards.MAX_NGRAM_ORDER", 3)
    build_shards(items, shard_dir, jobs=1)
    terms = load_shard(shard_path)
    # rebuilt at the new order: the 4- and 5-grams are gone, not inherited
    assert max(len(t) for t in terms) == 3

    key = _group_key([items[0][2]])
    monkeypatch.setattr("py3langid.train.shards.DOC_CAP", 3)
    assert _group_key([items[0][2]]) != key


def test_axis_first_appearance_order(tmp_path):
    """class column order is first appearance along the sorted walk, NOT
    alphabetical -- it fixes nb_classes, so pin it"""
    write_corpus(tmp_path, [(d, lang, b"some text here")
                            for d, langs in (("aaa", ["en", "fr"]), ("bbb", ["de", "en"]))
                            for lang in langs])
    items = list(walk_corpus(tmp_path))
    langs, dist, index = _axis(lang for _, lang, _ in items)
    assert langs == ["en", "fr", "de"]   # "de" is absent from the first domain
    assert dist.tolist() == [2, 1, 1]
    assert index == {"en": 0, "fr": 1, "de": 2}
    assert _axis(d for d, _, _ in items)[0] == ["aaa", "bbb"]


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
    feats = [b"ab", b"abc", b"bc", b"c", b"xy", b"aab", b"bca", b"cab"]
    rows, row_index, out = build_scanner(feats)
    stored = len(rows) // 256
    assert stored < len(out)  # sharing actually happened

    contents = {tuple(rows[(row_index[s] << 8):(row_index[s] << 8) + 256])
                for s in range(len(out))}
    assert len(contents) == stored


def test_feature_counts(tmp_path, monkeypatch):
    """NB numerators = one longest match per byte position, per language"""
    feats = [b"ab", b"abc", b"c"]
    rows, row_index, out = build_scanner(feats)

    docs = {"en": [b"abcabc", b"ab", b""], "fr": [b"cc", b"xabz"]}
    items = write_corpus(tmp_path, [("dom", lang, t) for lang, texts in docs.items() for t in texts])
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


def test_build_words_markers(tmp_path):
    """CJK characters near-exclusive to one CJK class (>= MARKER_MIN_DF docs) get a fixed credit"""
    yue = [("web", "yue", f"佢嘅書{i}".encode()) for i in range(W.MARKER_MIN_DF)]
    zh = [("web", "zh", f"他的書{i}".encode()) for i in range(W.MARKER_MIN_DF)]
    items, shard_dir = make_corpus(tmp_path, yue + zh + [("web", "en", b"book")])
    shards = build_shards(items, shard_dir, jobs=1)
    vocab, indptr, cols, vals = W.build_words(shards, {"yue": 0, "zh": 1, "en": 2})
    words = vocab.decode().split("\n")
    got = {words[i]: (int(cols[indptr[i]]), float(vals[indptr[i]]))
           for i in range(len(words)) if indptr[i + 1] > indptr[i]}
    credit = W.MARKER_WEIGHT
    assert got == {"佢": (0, credit), "嘅": (0, credit), "他": (1, credit), "的": (1, credit)}
    # shared 書 is no marker and CJK characters get no PMI credit; digits are not tokens
    assert "書" not in words and words == sorted(words)
    # fewer than MARKER_MIN_DF docs: no marker
    shards = build_shards(items[:2] + items[W.MARKER_MIN_DF:], str(tmp_path / "shards2"), jobs=1)
    vocab, *_ = W.build_words(shards, {"yue": 0, "zh": 1, "en": 2})
    assert "嘅" not in vocab.decode().split("\n")


def test_read_doc_matches_runtime_encoding(tmp_path):
    """training reads bytes exactly as the identifier normalizes them"""
    raw = 'Ünïcode MIXED café ЭТО'.encode() + 'é'.encode()[:1]  # cut mid-codepoint
    path = tmp_path / "doc.txt"
    path.write_bytes(raw)
    assert read_doc(path) == normalize(raw)
    assert read_doc(path)[0] == b'\xc3\xbcn\xc3\xafcode mixed caf\xc3\xa9 \xd1\x8d\xd1\x82\xd0\xbe'


def test_build_words(tmp_path, monkeypatch):
    """words kept by document frequency, long words need more docs, credits only where over-represented"""
    monkeypatch.setattr(W, "DOC_OFFSET", 0)
    docs = [("web", "en", b"the cat extraordinarily")] * 2 + [("web", "fr", b"le chat")] * 2 \
        + [("web", "fr", b"the rare rare")] + [("web", "de", b"der hund")] * 4
    items, shard_dir = make_corpus(tmp_path, docs)
    vocab, indptr, cols, vals = W.build_words(build_shards(items, shard_dir, jobs=1),
                                              {"en": 0, "fr": 1, "de": 2})
    words = vocab.decode().split("\n")
    assert words == ["cat", "chat", "der", "hund", "le", "the"]  # 'rare' twice in one doc, long word in two
    entries = {(words[i], int(c)) for i in range(len(words))
               for c in cols[indptr[i]:indptr[i + 1]]}
    assert entries == {("cat", 0), ("chat", 1), ("der", 2), ("hund", 2), ("le", 1), ("the", 0)}
    assert (vals >= W.CREDIT_FLOOR).all() and len(vals) == len(cols) == indptr[-1]


def test_min_df_by_utf8_length():
    assert [W._min_df("a" * n) for n in (8, 9, 14, 15)] == [2, 4, 4, 6]
    assert [W._min_df("é" * n) for n in (4, 5)] == [2, 4]  # 2 bytes per character


def test_build_words_marker_needs_owner_rate(tmp_path):
    """a character seen in MARKER_MIN_DF docs of one class and nowhere else is no marker
    when that class has many more docs (rate below MARKER_MIN_RATE)"""
    n = int(W.MARKER_MIN_DF / W.MARKER_MIN_RATE) + 1  # rare: MIN_DF docs out of n
    wuu = [("web", "wuu", f"侬好{i}".encode()) for i in range(n)]
    wuu += [("web", "wuu", f"溥儀{i}".encode()) for i in range(W.MARKER_MIN_DF)]
    zh = [("web", "zh", f"你好{i}".encode()) for i in range(W.MARKER_MIN_DF)]
    items, shard_dir = make_corpus(tmp_path, wuu + zh)
    vocab, *_ = W.build_words(build_shards(items, shard_dir, jobs=1), {"wuu": 0, "zh": 1})
    words = vocab.decode().split("\n")
    assert "侬" in words and "儀" not in words and "溥" not in words
