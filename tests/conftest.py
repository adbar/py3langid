import pytest


@pytest.fixture
def tokenize_order2(monkeypatch):
    """Tokenize byte order 2 only, so a shard payload is small enough to
    assert on exactly. Moves the tokenizer and the cache key together, but
    NOT selection: SELECT_ORDERS is derived at import, so tests needing a
    different selection pass ngram_select's `orders`."""
    monkeypatch.setattr("py3langid.train.shards.MAX_NGRAM_ORDER", 2)
