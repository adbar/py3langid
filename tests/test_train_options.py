"""Unit tests for ngram_select's order set and the DOC_CAP tokenization cap.

The n-gram orders, the DF pool size and the doc cap are constants, not
flags: tests that need a small term set patch MAX_NGRAM_ORDER or pass
`orders` explicitly, and tests that need a different cap patch DOC_CAP.
"""
import marshal

import pytest

from py3langid.train.shards import _build_shard, _group_key
from py3langid.train.stages import ngram_select


def test_ngram_select_orders():
    """`orders` alone decides which lengths are admissible"""
    doc_count = {b"a": 9, b"b": 8, b"ab": 7, b"abc": 6}
    assert ngram_select(doc_count, 10, orders={1, 2, 3}) == \
        [b"a", b"ab", b"abc", b"b"]
    assert ngram_select(doc_count, 10, orders={2, 3}) == [b"ab", b"abc"]
    assert ngram_select(doc_count, 10, orders={2}) == [b"ab"]
    assert ngram_select(doc_count, 10, orders=set()) == []


def test_ngram_select_default_orders():
    """the shipped order set: byte orders 2..5 plus the CJK bigram order"""
    from py3langid.train.common import (
        MAX_NGRAM_ORDER,
        MIN_NGRAM_ORDER,
        SELECT_ORDERS,
        TOKENIZE_ORDER,
    )
    assert SELECT_ORDERS == set(range(MIN_NGRAM_ORDER, MAX_NGRAM_ORDER + 1)) \
        | {TOKENIZE_ORDER}
    # a single byte is never selectable, so a 1-gram cannot become a feature
    assert ngram_select({b"a": 9, b"ab": 1}) == [b"ab"]


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
    with open(shard, "rb") as f:
        marshal.load(f)  # the cache key
        docfreq = marshal.load(f)
    # the bigrams of b"abcdef" are distinct: one doc each
    assert docfreq == dict.fromkeys(expected, 1)


def test_doc_cap_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """editing the cap must invalidate cached shards"""
    monkeypatch.setattr("py3langid.train.shards.DOC_CAP", 3)
    key3 = _group_key([str(tmp_path)])
    monkeypatch.setattr("py3langid.train.shards.DOC_CAP", 0)
    assert _group_key([str(tmp_path)]) != key3
