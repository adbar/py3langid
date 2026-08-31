"""Unit tests for the --doc_cap training option and ngram_select's order set.

The n-gram orders and the DF pool size are constants, not flags: tests that
need a small term set patch MAX_NGRAM_ORDER or pass `orders` explicitly.
"""
import marshal

from py3langid.train.common import set_shared
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


def _shard_counts(tmp_path, doc_cap, monkeypatch):
    """@returns the shard's docfreq dict, tokenized at order 2 only."""
    monkeypatch.setattr("py3langid.train.shards.MAX_NGRAM_ORDER", 2)
    doc = tmp_path / "doc0000.txt"
    doc.write_bytes(b"abcdef")
    shard = tmp_path / "shard"
    set_shared(doc_cap)
    _build_shard((str(shard), _group_key([str(doc)], doc_cap), [str(doc)]))
    with open(shard, "rb") as f:
        marshal.load(f)
        return marshal.load(f)


def test_doc_cap_truncates(tmp_path, monkeypatch):
    # cap 3 sees only b"abc", so b"cd" onwards is never tokenized
    assert _shard_counts(tmp_path, 3, monkeypatch) == {b"ab": 1, b"bc": 1}


def test_doc_cap_zero_reads_all(tmp_path, monkeypatch):
    assert set(_shard_counts(tmp_path, 0, monkeypatch)) == {
        b"ab", b"bc", b"cd", b"de", b"ef"}


def test_doc_cap_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """two caps must not read each other's shards"""
    assert _shard_counts(tmp_path, 3, monkeypatch) != \
        _shard_counts(tmp_path, 0, monkeypatch)
