"""Training pipeline constants and helpers."""

import multiprocessing as mp
import sys
import unicodedata
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from ..langid import decode_trimmed

MAX_NGRAM_ORDER = 5
MIN_NGRAM_ORDER = 2
DF_TOKENS = 60000        # candidate pool per order
FEATURES_PER_LANG = 1050 # per-language, not global (keeps script-novel langs viable)
DOC_CAP = 3000           # byte budget: gathering, tokenization, verifier, zxx
MIN_DOC = 500
MIN_DOMAINS = 2          # feature selection and topup both target this

CLUSTERS = (("ms", "id"), ("bs", "hr"), ("no", "nn", "da"),
            ("zh", "yue", "wuu"))
CLUSTER_K = 150  # extra features per cluster

TOKENIZE_ORDER = 6  # CJK codepoint bigrams (3+3 bytes)
SELECT_ORDERS = frozenset(range(MIN_NGRAM_ORDER, MAX_NGRAM_ORDER + 1)) \
    | {TOKENIZE_ORDER}


def is_cjk_bigram(term):
    """True if term is exactly two CJK codepoints (6 bytes)."""
    if len(term) != TOKENIZE_ORDER:
        return False
    try:
        s = term.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return len(s) == 2 and all(ord(ch) >= 0x2E80 for ch in s)


def latin_majority(doc):
    """True if doc has more Latin than Cyrillic letters."""
    text = doc.decode("utf-8", errors="surrogateescape")
    cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    lat = sum(1 for ch in text if ch.isalpha() and ch < "ɐ")
    return lat > cyr


class SplitScript(NamedTuple):
    """A language trained as two script-specific classes."""
    alt: str
    script: str
    alt_script: str
    routes_to_alt: Callable


SPLIT_SCRIPT = {
    "sr": SplitScript("srl", "Cyrl", "Latn", latin_majority),
    "uz": SplitScript("uzc", "Latn", "Cyrl", lambda doc: not latin_majority(doc)),
}

ALT_CLASS = {s.alt: lang for lang, s in SPLIT_SCRIPT.items()}
LABEL_ALIAS = {**ALT_CLASS, "nb": "no"}
CLASS_SCRIPT = {lang: s.script for lang, s in SPLIT_SCRIPT.items()}
CLASS_SCRIPT.update({s.alt: s.alt_script for s in SPLIT_SCRIPT.values()})


def route_script(out_dir, doc):
    """Route doc to its alt class dir if split-script."""
    spec = SPLIT_SCRIPT.get(out_dir.name)
    if spec and spec.routes_to_alt(doc):
        return out_dir.with_name(spec.alt)
    return out_dir


def script_filter(cls):
    "Doc predicate for `cls`'s script, or None if `cls` is not split-script."
    if cls in ALT_CLASS:
        return SPLIT_SCRIPT[ALT_CLASS[cls]].routes_to_alt
    if cls in SPLIT_SCRIPT:
        routes_to_alt = SPLIT_SCRIPT[cls].routes_to_alt
        return lambda doc: not routes_to_alt(doc)
    return None


def walk_corpus(root, skip_langs=(), pattern="*.txt"):
    """Yield (domain, lang, path) for docs three levels down, sorted."""
    for domain in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for lang_dir in sorted(p for p in domain.iterdir() if p.is_dir()):
            if lang_dir.name in skip_langs:
                continue
            for doc in sorted(lang_dir.glob(pattern)):
                if doc.is_file():
                    yield domain.name, lang_dir.name, str(doc)


def nfc_bytes(data):
    """NFC-normalize UTF-8 bytes, trimming partial trailing codepoints."""
    text = decode_trimmed(data)
    if text is None:
        return data
    return unicodedata.normalize("NFC", text).encode("utf-8")


def read_doc(path, cap=0):
    """Read NFC-normalized doc bytes, truncated to cap (0 = no cap)."""
    with open(path, "rb") as f:
        return nfc_bytes(f.read(cap) if cap else f.read())


def drop(corpus, paths):
    """Move docs to a sibling <corpus>_dropped tree, keeping relative paths."""
    dropped_root = Path(str(corpus).rstrip("/") + "_dropped")
    for p in paths:
        src = Path(p)
        dst = dropped_root / src.relative_to(corpus)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)


def job_count(processes=None):
    """Resolve to concrete worker count (None = all cores)."""
    return mp.cpu_count() if processes is None else max(1, processes)


def chunks(seq, size):
    """Split into chunks of at most size items."""
    size = max(1, size)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def job_chunks(seq, jobs):
    """One contiguous chunk per job."""
    return chunks(seq, -(-len(seq) // job_count(jobs)))


_SHARED = ()


def set_shared(*args):
    """MapPool initializer: stash per-worker constants."""
    global _SHARED
    _SHARED = args


def shared():
    return _SHARED


@contextmanager
def MapPool(processes=None, initializer=None, initargs=None, chunksize=1):
    """Process pool that falls back to serial map when processes=1."""
    processes = job_count(processes)

    if processes > 1:
        ctx = mp.get_context('fork') if sys.platform == 'darwin' else mp
        with ctx.Pool(processes, initializer, initargs) as pool:
            yield lambda fn, chunks: pool.imap_unordered(fn, chunks, chunksize)
    else:
        if initializer is not None:
            initializer(*initargs)
        yield map
