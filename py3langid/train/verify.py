"""Self-verify corpus filter.

Classifies every training doc with a verifier model and removes docs whose
prediction is a different, NON-confusable language (moved to a sibling
`<corpus>_dropped` tree, never deleted). Confusable groups are protected:
dropping there would launder the pair boundary through the verifier's bias.

`--paragraphs` filters at paragraph level instead (doc rewritten in place,
foreign paragraphs >= MIN_PARA bytes removed). Use an INDEPENDENT verifier
for the first (bootstrap) round on a fresh corpus — a model trained on the
corpus itself accepts the contamination it learned. After one clean round,
the retrained model is a valid self-verifier.

    python -m py3langid.train.verify --model MODEL_FILE CORPUS_DIR
"""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from .common import LABEL_ALIAS, MapPool, read_doc, walk_corpus

# Every pair within a group is protected; overlapping groups express
# near-cliques (kk/ky are NOT protected against each other, only vs tt/ba).
CONFUSABLE_GROUPS = [
    # South Slavic + Balkan
    {"bs", "hr", "sr"}, {"sr", "mk"},
    # Scandinavian
    {"no", "nn", "da"},
    # Malayo-Polynesian
    {"ms", "id", "ace"}, {"ace", "tl"}, {"bcl", "tl"},
    # Bantu
    {"xh", "zu", "sn", "st", "nso"}, {"lg", "sw"}, {"lg", "sn"},
    {"kik", "sn"}, {"kik", "sw"}, {"kik", "rw"},
    # Indic
    {"hi", "mr", "sa"}, {"hi", "ne", "sa"},
    {"gom", "mr"}, {"gom", "hi"}, {"gom", "ne"},
    # Turkic
    {"tt", "ba", "kk"}, {"tt", "ba", "ky"},
    {"uz", "tk", "az"}, {"uz", "tk", "tr"},
    {"crh", "tr"}, {"crh", "az"}, {"crh", "tt"},
    # Arabic + Persian + Indo-Aryan Perso-Arabic
    {"ar", "arz", "ary"}, {"fa", "ps"}, {"fa", "ar"},
    # Southern Uzbek (Perso-Arabic script)
    {"uzs", "fa"}, {"uzs", "ps"}, {"uzs", "ur"}, {"uzs", "ug"},
    # Chinese
    {"zh", "yue", "wuu"},
    # Romance + French creoles
    {"it", "lij", "vec"}, {"gcf", "gcr", "ht"}, {"gcf", "fr"},
    {"ext", "an"}, {"ext", "es"}, {"ext", "pt"},
    # Celtic / Germanic / Baltic
    {"gd", "ga"}, {"fy", "nl"}, {"fy", "af"}, {"ltg", "lv"},
    # historical vs modern
    {"grc", "el"}, {"hbo", "he"},
    # other
    {"pcm", "en"}, {"fuv", "ha"}, {"fuv", "om"},
]
CONFUSABLE = {frozenset(p) for g in CONFUSABLE_GROUPS
              for p in combinations(sorted(g), 2)}
MIN_PARA = 150
MIN_DOC = 500
MAX_CLASSIFY = 3000

_ident = None


def _init(model_path):
    global _ident
    from py3langid.langid import LanguageIdentifier
    _ident = LanguageIdentifier.from_modelpath(model_path)


def is_foreign(label, pred):
    label = LABEL_ALIAS.get(label, label)
    pred = LABEL_ALIAS.get(pred, pred)
    return pred != label and frozenset((label, pred)) not in CONFUSABLE


def _check_doc(arg):
    lang, path = arg
    pred, _ = _ident.classify(read_doc(path, MAX_CLASSIFY))
    return path if is_foreign(lang, pred) else None


def _filter_paragraphs(arg):
    """@returns (lang, path, stripped bytes, whole doc now junk)."""
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
        return lang, path, stripped, True  # marker: whole doc now junk
    Path(path).write_bytes(out)
    return lang, path, stripped, False


def corpus_items(corpus, verifier_langs=None):
    items = []
    skipped_langs = set()
    # zxx is synthetic by construction: no verifier can vouch for it
    for _domain, lang, path in walk_corpus(corpus, skip_langs=("zxx",)):
        label = LABEL_ALIAS.get(lang, lang)
        # bootstrap: skip langs the verifier doesn't know (it would
        # drop 100% of their docs as foreign)
        if verifier_langs is not None and label not in verifier_langs:
            skipped_langs.add(lang)
            continue
        items.append((lang, path))
    if skipped_langs:
        print(f"verify: skipped {len(skipped_langs)} unknown lang(s): {sorted(skipped_langs)}")
    return items


def drop(corpus, paths):
    dropped_root = Path(str(corpus).rstrip("/") + "_dropped")
    for p in paths:
        src = Path(p)
        dst = dropped_root / src.relative_to(corpus)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="verifier model file (npz.xz)")
    parser.add_argument("--paragraphs", action="store_true",
                        help="filter foreign paragraphs instead of whole docs")
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument("corpus", metavar="CORPUS_DIR")
    args = parser.parse_args(argv)

    # load verifier's known languages to skip unknown lang dirs
    from py3langid.langid import LanguageIdentifier
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
