"""Train a langid model from a prepared corpus."""

import argparse
import multiprocessing as mp
import os
from collections import Counter

import numpy as np

from ..modelio import Model, save_model
from .common import ALT_CLASS, COUNT_FLOOR, FEATURES_PER_LANG, walk_corpus
from .scanner import build_scanner
from .shards import build_shards, count_matrices, merge_docfreq
from .stages import compute_IG, feature_counts, ngram_select, select_LD_features
from .words import build_words


def _axis(values):
    """(names in first-appearance order, count array, name→column) of *values*."""
    counts = Counter(values)
    return list(counts), np.array(list(counts.values())), {n: i for i, n in enumerate(counts)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model", help="save output to MODEL_DIR", metavar="MODEL_DIR")
    parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no parallelization)")
    parser.add_argument("--feats_per_lang", type=int, metavar='N', help="select top N features for each language", default=FEATURES_PER_LANG)
    parser.add_argument("--shards", metavar="SHARD_DIR", help="n-gram count shard cache (default: CORPUS_DIR.shards)")
    parser.add_argument("corpus", help="read corpus from CORPUS_DIR", metavar="CORPUS_DIR")

    args = parser.parse_args(argv)

    if args.jobs is None:
        args.jobs = min(10, mp.cpu_count())

    model_dir = args.model or os.path.join('.', os.path.basename(args.corpus) + '.model')

    os.makedirs(model_dir, exist_ok=True)

    print("corpus path:", args.corpus)
    print("model path:", model_dir)

    items = list(walk_corpus(args.corpus))
    langs, lang_dist, lang_index = _axis(lang for _, lang, _ in items)
    domains, domain_dist, domain_index = _axis(d for d, _, _ in items)

    def _summary(names, dist):
        return f"({len(names)}): " + ' '.join(
            f"{n}({c})" for n, c in zip(names, dist))

    print("langs" + _summary(langs, lang_dist))
    print("domains" + _summary(domains, domain_dist))
    print(f"identified {len(items)} files")

    shard_dir = args.shards or os.path.normpath(args.corpus) + '.shards'
    shard_items = build_shards(items, shard_dir, args.jobs)

    doc_count = merge_docfreq(shard_items, args.jobs)
    print(f"tallied document frequency of {len(doc_count)} terms")

    features = ngram_select(doc_count)
    del doc_count
    print(f"selected {len(features)} DF features")

    cm_lang, cm_domain = count_matrices(
        shard_items, features, lang_index, domain_index, args.jobs)

    nonempty = cm_lang.any(0)
    empty = sorted(lang for lang, j in lang_index.items() if not nonempty[j])
    if empty:
        parser.error(f"no n-grams for {len(empty)} class(es): {empty} "
                     "-- check for empty or unreadable docs")

    print("computing information gain")
    domain_ig = compute_IG(cm_domain, domain_dist)
    LDidx = select_LD_features(cm_lang, lang_dist, domain_ig, args.feats_per_lang)
    LDfeats = sorted(features[i] for i in LDidx)
    print(f'selected {len(LDfeats)} features')

    tk_nextmove, tk_row, tk_output = build_scanner(LDfeats)
    emitting = sum(f >= 0 for f in tk_output)
    print(f"scanner: {len(tk_output)} states, {emitting} emitting, "
          f"{len(tk_nextmove) // 256} distinct transition rows")

    nb_classes = [ALT_CLASS.get(lang, lang) for lang in langs]
    nb_pc = np.log(lang_dist)

    print("counting longest-match feature occurrences")
    prod = feature_counts(items, tk_nextmove, tk_row, tk_output, len(LDfeats),
                          lang_index, args.jobs)
    prod[prod <= COUNT_FLOOR] = 0  # thin evidence is noise and bulk
    nb_ptc = np.log(1.0 + prod) - np.log(len(LDfeats) + prod.sum(0))  # add-one smoothed

    words = build_words(shard_items, lang_index)
    print(f"word table: {len(words.indptr) - 1} tokens, {len(words.cols)} entries")

    model = Model(nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output, words)
    npz_path = os.path.join(model_dir, 'model.npz.xz')
    save_model(npz_path, model)
    print(f"wrote model to {npz_path} ({os.path.getsize(npz_path)} bytes)")


if __name__ == "__main__":
    main()
