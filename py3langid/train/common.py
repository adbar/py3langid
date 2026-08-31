"""
Common constants and helpers for the training pipeline.
"""

import multiprocessing as mp
import sys
import unicodedata
from collections.abc import Callable
from contextlib import closing, contextmanager
from pathlib import Path
from typing import NamedTuple

# Pipeline defaults = the adopted release config (armE, 2026-08-27;
# FEATURES_PER_LANG raised to 1050 on 2026-08-31), overridable via
# command-line options
MAX_NGRAM_ORDER = 5 # largest order of n-grams to consider
MIN_NGRAM_ORDER = 2 # smallest order of n-grams to consider
DF_TOKENS = 60000 # candidate pool: top tokens by document frequency, per order
# Number of features per language. Raised 900 -> 1050 after a measured
# sweep (k=950..1400): short input (<120 B) gains and long input (>=200 B)
# loses, both monotonically in k, and 1050 is the pooled optimum. Keep this
# PER-LANGUAGE rather than capping the total: at 142 languages the union
# shares 33% of picks, so a global cap would make each new language shrink
# every existing language's budget (~20 new languages would undo this raise)
# and would starve exactly the script-novel additions, which share nothing.
FEATURES_PER_LANG = 1050
DOC_CAP = 3000 # bytes per doc at tokenization; equalizes doc byte weight

# Confusable clusters get CLUSTER_K extra features each, chosen by
# cluster-restricted IG from features the per-language quota missed.
CLUSTERS = (("ms", "id"), ("bs", "hr"), ("no", "nn", "da"),
            ("zh", "yue", "wuu"))
CLUSTER_K = 150

# Extra candidate pool: byte order 5 cannot span two 3-byte codepoints, so
# CJK codepoint bigrams are 6-byte n-grams (hence TOKENIZE_ORDER) offered to
# the clusters alongside the DF pool. They are the only use the pipeline has
# for that order, so tokenization emits nothing else there.
CJK_DF_FLOOR = 20
TOKENIZE_ORDER = 6


def is_cjk_bigram(term):
    """True for a 6-byte n-gram encoding exactly two CJK codepoints."""
    if len(term) != 6:
        return False
    try:
        s = term.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return len(s) == 2 and all(ord(ch) >= 0x2E80 for ch in s)


def latin_majority(doc):
    """True if a doc (bytes) has more Latin than Cyrillic letters."""
    text = doc.decode("utf-8", errors="surrogateescape")
    cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    lat = sum(1 for ch in text if ch.isalpha() and ch < "ɐ")
    return lat > cyr


class SplitScript(NamedTuple):
    """A language written in two scripts, trained as two classes. Adding
    one means adding one entry; the tables below derive from it."""
    alt: str                        # extra class dir
    script: str                     # ISO 15924 script of the base class
    alt_script: str                 # ISO 15924 script of the alt class
    routes_to_alt: Callable         # True if a doc belongs to `alt`


SPLIT_SCRIPT = {
    "sr": SplitScript("srl", "Cyrl", "Latn", latin_majority),
    "uz": SplitScript("uzc", "Latn", "Cyrl", lambda doc: not latin_majority(doc)),
}

# alt class dir -> base language
ALT_CLASS = {s.alt: lang for lang, s in SPLIT_SCRIPT.items()}
# internal script-split class dirs and legacy labels -> canonical label
LABEL_ALIAS = {**ALT_CLASS, "nb": "no"}
# ISO 15924 script per split-script class, for sources with per-script configs
CLASS_SCRIPT = {lang: s.script for lang, s in SPLIT_SCRIPT.items()}
CLASS_SCRIPT.update({s.alt: s.alt_script for s in SPLIT_SCRIPT.values()})


def route_script(out_dir, doc):
    """Route a split-script doc to its alt class directory."""
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
    """The pipeline's one definition of a training doc: a `pattern` file
    exactly three levels down, sorted. Stray files and nested dirs are
    ignored, so neither can become a phantom class.

    @returns (domain, lang, path) string triples
    """
    for domain in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for lang_dir in sorted(p for p in domain.iterdir() if p.is_dir()):
            if lang_dir.name in skip_langs:
                continue
            for doc in sorted(lang_dir.glob(pattern)):
                if doc.is_file():
                    yield domain.name, lang_dir.name, str(doc)


def nfc_bytes(data):
    """NFC-normalize UTF-8 bytes; undecodable data is returned unchanged
    (up to 3 trailing bytes are dropped to recover from cap truncation)."""
    for trim in range(4):
        chunk = data[:len(data) - trim] if trim else data
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            continue
        return unicodedata.normalize("NFC", text).encode("utf-8")
    return data


def read_doc(path, cap=0):
    """Read a training doc's NFC-normalized bytes, truncated to cap bytes
    (0 = no cap)."""
    with open(path, "rb") as f:
        return nfc_bytes(f.read(cap) if cap else f.read())


def job_count(processes=None):
    """Resolve a jobs setting to a concrete worker count (None = all cores)."""
    return mp.cpu_count() if processes is None else max(1, processes)


@contextmanager
def MapPool(processes=None, initializer=None, initargs=None, chunksize=1):
    """
    Contextmanager to express the common pattern of not using multiprocessing if
    only 1 job is allocated (for example for debugging reasons).
    Results are lazy and unordered: consume them inside the with-block.
    """
    processes = job_count(processes)

    if processes > 1:
        # macOS defaults to 'spawn' since Python 3.8, which breaks the
        # initializer-based global sharing the pipeline relies on; a local
        # context avoids changing the process-wide default on import.
        ctx = mp.get_context('fork') if sys.platform == 'darwin' else mp
        with closing(ctx.Pool(processes, initializer, initargs)) as pool:
            yield lambda fn, chunks: pool.imap_unordered(fn, chunks, chunksize)
        pool.join()
    else:
        if initializer is not None:
            initializer(*initargs)
        yield map
