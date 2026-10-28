import functools
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import py3langid.langid as mod
from py3langid.modelio import WordTable

NO_WORDS = WordTable(
    b"", np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
)


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


def longest_counts(feats, data):
    """reference scanner, naive O(n*|feats|): counts of the longest feature ending at each byte"""
    hits = (
        max((f for f in feats if data[:end].endswith(f)), key=len, default=None)
        for end in range(1, len(data) + 1)
    )
    return Counter(feats.index(h) for h in hits if h is not None)


@pytest.fixture
def tokenize_order2(monkeypatch):
    """Tokenize byte order 2 only, so a shard payload is small enough to
    assert on exactly. Moves the tokenizer and the cache key together."""
    monkeypatch.setattr("py3langid.train.shards.MAX_NGRAM_ORDER", 2)


@pytest.fixture(scope="session", autouse=True)
def _shipped_model_loaded_once():
    """every load of the shipped model reuses one read: Model arrays are never mutated"""
    real, shipped = mod.load_model, mod.MODEL_FILE.resolve()
    cached = functools.cache(lambda: real(shipped))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            mod,
            "load_model",
            lambda path: cached() if Path(path).resolve() == shipped else real(path),
        )
        yield
