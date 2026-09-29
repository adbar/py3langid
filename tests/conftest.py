import numpy as np
import pytest

from py3langid.modelio import WordTable

NO_WORDS = WordTable(b"", np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32),
                     np.zeros(0, dtype=np.float32))


def write_corpus(root, docs):
    """root/<domain>/<lang>/docNNNN.txt per (domain, lang, data). Returns walk_corpus items."""
    items = []
    for domain, lang, data in docs:
        cell = root / domain / lang
        cell.mkdir(parents=True, exist_ok=True)
        path = cell / f"doc{sum(i[:2] == (domain, lang) for i in items):04d}.txt"
        path.write_bytes(data.encode("utf-8", "surrogateescape") if isinstance(data, str) else data)
        items.append((domain, lang, str(path)))
    return items


@pytest.fixture
def tokenize_order2(monkeypatch):
    """Tokenize byte order 2 only, so a shard payload is small enough to
    assert on exactly. Moves the tokenizer and the cache key together."""
    monkeypatch.setattr("py3langid.train.shards.MAX_NGRAM_ORDER", 2)
