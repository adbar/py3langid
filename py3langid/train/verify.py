"""Corpus verifier: drop docs predicted as a different non-confusable language."""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from ..langid import LanguageIdentifier
from .common import (
    DOC_CAP,
    LABEL_ALIAS,
    MIN_DOC,
    MapPool,
    drop,
    read_doc,
    walk_corpus,
)

CONFUSABLE_GROUPS = [
    {"bs", "hr", "sr"}, {"sr", "mk"},
    {"no", "nn", "da"},
    {"ms", "id", "ace"}, {"ace", "tl"}, {"bcl", "tl"},
    {"xh", "zu", "sn", "st", "nso"}, {"lg", "sw"}, {"lg", "sn"},
    {"kik", "sn"}, {"kik", "sw"}, {"kik", "rw"},
    {"hi", "mr", "sa"}, {"hi", "ne", "sa"},
    {"gom", "mr"}, {"gom", "hi"}, {"gom", "ne"},
    {"tt", "ba", "kk"}, {"tt", "ba", "ky"},
    {"uz", "tk", "az"}, {"uz", "tk", "tr"},
    {"crh", "tr"}, {"crh", "az"}, {"crh", "tt"},
    {"ar", "arz", "ary"}, {"fa", "ps"}, {"fa", "ar"},
    {"uzs", "fa"}, {"uzs", "ps"}, {"uzs", "ur"}, {"uzs", "ug"},
    {"zh", "yue", "wuu"},
    {"it", "lij", "vec"}, {"gcf", "gcr", "ht"}, {"gcf", "fr"},
    {"ext", "an"}, {"ext", "es"}, {"ext", "pt"},
    {"gd", "ga"}, {"fy", "nl"}, {"fy", "af"}, {"ltg", "lv"},
    {"grc", "el"}, {"hbo", "he"},
    {"pcm", "en"}, {"fuv", "ha"}, {"fuv", "om"},
]
CONFUSABLE = {frozenset(p) for g in CONFUSABLE_GROUPS
              for p in combinations(sorted(g), 2)}
MIN_PARA = 150

_ident = None


def _init(model_path):
    global _ident
    _ident = LanguageIdentifier.from_modelpath(model_path)


def is_foreign(label, pred):
    label = LABEL_ALIAS.get(label, label)
    pred = LABEL_ALIAS.get(pred, pred)
    return pred != label and frozenset((label, pred)) not in CONFUSABLE


def _check_doc(arg):
    lang, path = arg
    pred, _ = _ident.classify(read_doc(path, DOC_CAP))
    return path if is_foreign(lang, pred) else None


def _filter_paragraphs(arg):
    lang, path = arg
    data = Path(path).read_bytes()
    kept, stripped = [], 0
    for para in data.split(b"\n"):
        if len(para) >= MIN_PARA and is_foreign(lang, _ident.classify(para)[0]):
            stripped += len(para)
            continue
        kept.append(para)
    if not stripped:
        return lang, path, 0, False
    out = b"\n".join(kept)
    if len(out) < MIN_DOC:
        return lang, path, stripped, True
    Path(path).write_bytes(out)
    return lang, path, stripped, False


def corpus_items(corpus, verifier_langs=None):
    items = []
    skipped_langs = set()
    for _domain, lang, path in walk_corpus(corpus, skip_langs=("zxx",)):
        label = LABEL_ALIAS.get(lang, lang)
        if verifier_langs is not None and label not in verifier_langs:
            skipped_langs.add(lang)
            continue
        items.append((lang, path))
    if skipped_langs:
        print(f"verify: skipped {len(skipped_langs)} unknown lang(s): {sorted(skipped_langs)}")
    return items


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="verifier model file (npz.xz)")
    parser.add_argument("--paragraphs", action="store_true",
                        help="filter foreign paragraphs instead of whole docs")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="parallel workers (default: 8)")
    parser.add_argument("corpus", metavar="CORPUS_DIR")
    args = parser.parse_args(argv)

    verifier_langs = set(LanguageIdentifier.from_modelpath(args.model).nb_classes)
    items = corpus_items(args.corpus, verifier_langs)
    if args.paragraphs:
        stripped = Counter()
        junk = []
        with MapPool(args.jobs, _init, (args.model,), chunksize=200) as f:
            for lang, path, n, dead in f(_filter_paragraphs, items):
                if n:
                    stripped[lang] += n
                if dead:
                    junk.append(path)
        drop(args.corpus, junk)
        print("stripped bytes by lang:", stripped.most_common(15))
        print(f"docs dropped entirely: {len(junk)}")
    else:
        with MapPool(args.jobs, _init, (args.model,), chunksize=500) as f:
            drops = [p for p in f(_check_doc, items) if p]
        drop(args.corpus, drops)
        by_lang = Counter(Path(p).parent.name for p in drops)
        print(f"dropped {len(drops)} of {len(items)} docs:", by_lang.most_common(15))


if __name__ == "__main__":
    main()
