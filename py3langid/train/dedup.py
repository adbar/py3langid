"""Cross-domain line dedup per language, URLs and digits masked so templated stubs count
as one (first occurrence kept, blank lines kept). Docs left without text are dropped."""
import re
from collections import defaultdict
from pathlib import Path

from .common import drop, walk_corpus

DIGITS = re.compile(rb"\d+")
URL = re.compile(rb"https?://\S+")


def dedup(corpus):
    """Returns lines removed."""
    seen = defaultdict(set)
    removed, empty = 0, []
    for _domain, lang, path in walk_corpus(corpus, skip_langs=("zxx",)):
        lines = Path(path).read_bytes().split(b"\n")
        kept = []
        for ln in lines:
            key = DIGITS.sub(b"0", URL.sub(b"U", ln))
            if ln and key in seen[lang]:
                continue
            seen[lang].add(key)
            kept.append(ln)
        if not any(kept):
            empty.append(path)
        elif len(kept) < len(lines):
            Path(path).write_bytes(b"\n".join(kept))
        removed += len(lines) - len(kept)
    drop(corpus, empty)
    return removed

