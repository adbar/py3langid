"""Unit tests for ngram_select's per-order pools and the DOC_CAP tokenization cap.

The n-gram orders, the DF pool size and the doc cap are constants, not
flags: tests that need a small term set patch MAX_NGRAM_ORDER, and tests
that need a different cap patch DOC_CAP.
"""


import pytest

from py3langid.train.shards import _build_shard, _group_key, load_shard
from py3langid.train.stages import ngram_select


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
    docfreq = load_shard(str(shard))
    # the bigrams of b"abcdef" are distinct: one doc each
    assert docfreq == dict.fromkeys(expected, 1)


def test_doc_cap_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """editing the cap must invalidate cached shards"""
    monkeypatch.setattr("py3langid.train.shards.DOC_CAP", 3)
    key3 = _group_key([str(tmp_path)])
    monkeypatch.setattr("py3langid.train.shards.DOC_CAP", 0)
    assert _group_key([str(tmp_path)]) != key3
