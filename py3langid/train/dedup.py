"""Cross-domain exact line dedup per language (first occurrence kept)."""
import sys
from collections import defaultdict
from pathlib import Path

from .common import MIN_DOC, drop, walk_corpus

MIN_LINE = 60


def dedup(corpus):
    """Returns (lines_removed, docs_rewritten, docs_dropped)."""
    removed = touched = 0
    dropped = []
    docs_by_lang = defaultdict(list)
    for _domain, lang, path in walk_corpus(corpus, skip_langs=("zxx",)):
        docs_by_lang[lang].append(Path(path))
    for docs in docs_by_lang.values():
        s = set()
        for doc in docs:
            lines = doc.read_bytes().split(b"\n")
            new = set()
            kept = []
            for ln in lines:
                if len(ln) >= MIN_LINE:
                    if ln in s or ln in new:
                        removed += 1
                        continue
                    new.add(ln)
                kept.append(ln)
            if len(kept) != len(lines):
                out = b"\n".join(kept)
                if len(out.strip()) < MIN_DOC:  # too little left to train on
                    dropped.append(doc)
                    continue  # a dropped doc's lines must stay usable elsewhere
                doc.write_bytes(out)
                touched += 1
            s |= new
    drop(corpus, dropped)
    return removed, touched, len(dropped)


if __name__ == "__main__":
    removed, touched, dropped = dedup(sys.argv[1])
    print(f"dedup: removed {removed} duplicate lines across {touched} docs, "
          f"dropped {dropped} docs left under {MIN_DOC} bytes")
