"""Unit tests for the numeric core of the training pipeline."""
import math
import os

import numpy as np

from py3langid.train.shards import build_shards, count_ngrams, load_shard
from py3langid.train.stages import (
    compute_IG,
    compute_IG_binarized,
    entropy,
    ngram_select,
    select_LD_features,
)


def test_entropy():
    assert entropy([1, 1]) == np.log(2)
    assert entropy([2, 0]) == 0.0
    assert entropy([1, 1, 1, 1]) == np.log(4)


def test_count_ngrams():
    assert count_ngrams(b"abab", 2) == {b"a": 2, b"b": 2, b"ab": 2, b"ba": 1}
    assert count_ngrams(b"", 2) == {}


def test_compute_IG_nonbinarized():
    # 2 events with 2 docs each. b'aa' occurs only in event 0 (perfectly
    # discriminative, IG = log 2); b'bb' occurs once per event (IG = 0).
    cm = np.array([[2, 0], [1, 1]])
    dist = np.array([2, 2])
    ig = compute_IG(cm, dist)
    assert math.isclose(ig[0], math.log(2))
    assert math.isclose(ig[1], 0.0, abs_tol=1e-12)


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


def test_select_LD_features():
    # LD = IG_lang - IG_domain: term 2 is penalized for being domain-informative
    ld = np.array([
        [0.7, 0.1],
        [0.1, 0.6],
        [-0.1, -0.1],
    ])
    assert select_LD_features(ld, 1) == {0, 1}
    assert select_LD_features(ld, 3) == {0, 1, 2}


def test_ngram_select():
    doc_count = {b"a": 5, b"b": 3, b"ab": 10, b"cd": 1}
    feats = ngram_select(doc_count, max_order=2, tokens_per_order=1, min_order=1)
    assert feats == [b"a", b"ab"]


def test_build_shards_cache(tmp_path):
    lang_dir = tmp_path / "corpus" / "web" / "en"
    lang_dir.mkdir(parents=True)
    doc0 = lang_dir / "doc0.txt"
    doc0.write_bytes(b"abab")
    (lang_dir / "doc1.txt").write_bytes(b"ab")
    items = [("web", "en", str(lang_dir / f"doc{i}.txt")) for i in range(2)]
    shard_dir = str(tmp_path / "shards")

    [(domain, lang, shard_path)] = build_shards(items, shard_dir, 2, jobs=1)
    assert (domain, lang) == ("web", "en")
    docfreq, totalfreq = load_shard(shard_path)
    assert docfreq[b"ab"] == 2
    assert totalfreq[b"ab"] == 3
    assert totalfreq[b"ba"] == 1

    # unchanged corpus: shard is reused, not rewritten
    mtime = os.path.getmtime(shard_path)
    build_shards(items, shard_dir, 2, jobs=1)
    assert os.path.getmtime(shard_path) == mtime

    # lower requested order is served by the cached higher-order shard
    build_shards(items, shard_dir, 1, jobs=1)
    assert os.path.getmtime(shard_path) == mtime

    # changed doc invalidates and rebuilds the shard
    doc0.write_bytes(b"zzzz")
    build_shards(items, shard_dir, 2, jobs=1)
    docfreq, totalfreq = load_shard(shard_path)
    assert b"ab" in docfreq and docfreq[b"ab"] == 1
    assert totalfreq[b"zz"] == 3
