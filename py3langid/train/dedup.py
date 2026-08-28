"""Exact line dedup: remove repeated lines (>= MIN_LINE bytes) per language,
cross-domain, first occurrence kept. Deterministic (sorted traversal) and
idempotent. Measured accuracy-neutral; kept for signal hygiene — duplicate
boilerplate inflates document frequencies.

    python -m py3langid.train.dedup CORPUS_DIR
"""
import sys
from collections import defaultdict
from pathlib import Path

from .common import walk_corpus

MIN_LINE = 60


def dedup(corpus):
    """@returns (lines removed, docs rewritten)."""
    seen = defaultdict(set)
    removed = touched = 0
    docs_by_lang = defaultdict(list)
    # zxx is generated deterministically — leave it byte-exact
    for _domain, lang, path in walk_corpus(corpus, skip_langs=("zxx",)):
        docs_by_lang[lang].append(Path(path))
    for lang, docs in docs_by_lang.items():
        s = seen[lang]
        for doc in docs:
            lines = doc.read_bytes().split(b"\n")
            kept = []
            for ln in lines:
                if len(ln) >= MIN_LINE:
                    if ln in s:
                        removed += 1
                        continue
                    s.add(ln)
                kept.append(ln)
            if len(kept) != len(lines):
                touched += 1
                doc.write_bytes(b"\n".join(kept))
    return removed, touched


if __name__ == "__main__":
    removed, touched = dedup(sys.argv[1])
    print(f"dedup: removed {removed} duplicate lines across {touched} docs")
