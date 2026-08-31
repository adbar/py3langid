"""Unit tests for the --min_order and --doc_cap training options."""
import marshal

from py3langid.train.shards import _build_shard, _group_key, _setup_build
from py3langid.train.stages import ngram_select


def test_ngram_select_min_order():
    doc_count = {b"a": 9, b"b": 8, b"ab": 7, b"abc": 6}
    assert ngram_select(doc_count, max_order=3, tokens_per_order=10, min_order=1) == \
        [b"a", b"ab", b"abc", b"b"]
    assert ngram_select(doc_count, max_order=3, tokens_per_order=10, min_order=2) == \
        [b"ab", b"abc"]
    assert ngram_select(doc_count, max_order=2, tokens_per_order=10, min_order=2) == [b"ab"]


def _shard_counts(tmp_path, doc_cap):
    """@returns the shard's docfreq dict (order 2, the lowest counted)."""
    doc = tmp_path / "doc0000.txt"
    doc.write_bytes(b"abcdef")
    shard = tmp_path / "shard"
    _setup_build(2, doc_cap)
    _build_shard((str(shard), _group_key([str(doc)], doc_cap), [str(doc)]))
    with open(shard, "rb") as f:
        marshal.load(f)
        return marshal.load(f)


def test_doc_cap_truncates(tmp_path):
    # cap 3 sees only b"abc", so b"cd" onwards is never tokenized
    assert _shard_counts(tmp_path, doc_cap=3) == {b"ab": 1, b"bc": 1}


def test_doc_cap_zero_reads_all(tmp_path):
    assert set(_shard_counts(tmp_path, doc_cap=0)) == {
        b"ab", b"bc", b"cd", b"de", b"ef"}
