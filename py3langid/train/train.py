"""
train.py -
All-in-one tool for easy training of a model for langid.py.
"""

import argparse
import multiprocessing as mp
import os
from collections import Counter

import numpy as np

from ..modelio import save_model
from .common import (
    CLUSTER_K,
    CLUSTERS,
    DOC_CAP,
    FEATURES_PER_LANG,
    LABEL_ALIAS,
)
from .scanner import build_scanner
from .shards import build_shards, count_matrices, merge_docfreq
from .stages import (
    cluster_features,
    compute_IG,
    compute_IG_binarized,
    feature_counts,
    index_corpus,
    ngram_select,
    select_LD_features,
)


def _axis(names, values):
    """One class axis (languages or domains), in `names` order.
    @returns (per-name doc counts as an array, name -> column)"""
    counts = Counter(values)
    return (np.array([counts[name] for name in names]),
            {name: i for i, name in enumerate(names)})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model", help="save output to MODEL_DIR", metavar="MODEL_DIR")
    parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no parallelization)")
    parser.add_argument("--doc_cap", type=int, default=DOC_CAP,
        help="truncate each doc to N bytes at tokenization (0 = no cap)")
    parser.add_argument("--feats_per_lang", type=int, metavar='N', help="select top N features for each language", default=FEATURES_PER_LANG)
    parser.add_argument("--shards", metavar="SHARD_DIR", help="n-gram count shard cache (default: CORPUS_DIR.shards)")
    parser.add_argument("corpus", help="read corpus from CORPUS_DIR", metavar="CORPUS_DIR")

    args = parser.parse_args(argv)

    if args.jobs is None:
        args.jobs = min(10, mp.cpu_count())

    corpus_name = os.path.basename(args.corpus)
    if args.model:
        model_dir = args.model
    else:
        model_dir = os.path.join('.', corpus_name+'.model')

    os.makedirs(model_dir, exist_ok=True)

    # display paths
    print("corpus path:", args.corpus)
    print("model path:", model_dir)

    # first-appearance walk order fixes the class column order
    items, langs, domains = index_corpus(args.corpus)
    lang_dist, lang_index = _axis(langs, (lang for _, lang, _ in items))
    domain_dist, domain_index = _axis(domains, (d for d, _, _ in items))

    def _summary(names, dist):
        return f"({len(names)}): " + ' '.join(
            f"{n}({c})" for n, c in zip(names, dist))

    print("langs" + _summary(langs, lang_dist))
    print("domains" + _summary(domains, domain_dist))
    print(f"identified {len(items)} files")

    # Tokenize the corpus into per-(domain, lang) n-gram count shards.
    # Cached shards are reused; only changed directories are re-tokenized.
    shard_dir = args.shards or os.path.normpath(args.corpus) + '.shards'
    shard_items = build_shards(items, shard_dir, args.jobs, args.doc_cap)

    doc_count = merge_docfreq(shard_items, args.jobs)
    print(f"tallied document frequency of {len(doc_count)} terms")

    # One pool: top DF_TOKENS per admissible order, CJK bigrams among them,
    # so CJK evidence competes for quotas like any n-gram.
    features = ngram_select(doc_count)
    doc_count = None
    print(f"selected {len(features)} DF features")

    # One shard pass for the whole IG stage: per-lang and per-domain document
    # frequency plus each term's per-lang domain presence. The NB numerators
    # are NOT counted here -- they come from the scanner's longest-match
    # emission below, so shard n-gram totals are never needed.
    cm_lang, cm_domain, domcount = count_matrices(
        shard_items, features, lang_index, domain_index, args.jobs)

    # a class with no counts would ship as a uniform, unlearnable label
    nonempty = cm_lang.any(0)
    empty = sorted(lang for lang, j in lang_index.items() if not nonempty[j])
    if empty:
        parser.error(f"no n-grams for {len(empty)} class(es): {empty} "
                     "-- check for empty or unreadable docs")

    # Select features by LD weight (per-language IG minus domain IG),
    # candidates restricted to terms seen in >=2 of the language's domains
    # (or, for a single-domain language, in its one domain)
    print("computing information gain")
    ld = (compute_IG_binarized(cm_lang, lang_dist)
          - compute_IG(cm_domain, domain_dist)[:, None])
    # one shard per (domain, lang) dir, so a lang's domains are a shard tally
    shards_per_lang = Counter(lang for _, lang, _ in shard_items)
    need = np.array([min(2, shards_per_lang[lang]) for lang in langs])
    present = domcount >= need[None, :]
    LDidx = select_LD_features(ld, args.feats_per_lang, present)
    extra = cluster_features(cm_lang, lang_dist, lang_index, features, LDidx,
                             CLUSTERS, CLUSTER_K)
    print(f"added {len(extra)} cluster features")
    LDidx |= extra
    # term order fixes the scanner's feature indices, hence nb_ptc's rows
    LDfeats = sorted(features[i] for i in LDidx)
    print(f'selected {len(LDfeats)} features')

    # Compile a scanner for the LDfeats (one longest match per position)
    tk_nextmove, tk_row, tk_output = build_scanner(LDfeats)
    emitting = sum(f >= 0 for f in tk_output)
    print(f"scanner: {len(tk_output)} states, {emitting} emitting, "
          f"{len(tk_nextmove) // 256} distinct transition rows")

    # Assemble the NB model (duplicate labels are fine:
    # classification returns nb_classes[argmax])
    nb_classes = [LABEL_ALIAS.get(lang, lang) for lang in langs]
    # log P(class) from the per-class doc counts. The old --prior_cap 1200
    # was measured to be a no-op (counts run 135..1318, so it clipped four
    # classes by <=0.09 nats) and is gone: bare defaults now reproduce the
    # release. The priors themselves DO earn their place -- dropping them
    # costs CommonLID -519 labels (p=2e-61). Every lang_dist entry is >= 1 by
    # construction, so the log is finite.
    nb_pc = np.log(lang_dist)

    # NB numerators: count what the runtime accumulates, i.e. one longest
    # match per byte position, rather than every n-gram occurrence
    print("counting longest-match feature occurrences")
    prod = feature_counts(items, tk_nextmove, tk_row, tk_output, len(LDfeats),
                          lang_index, args.doc_cap, args.jobs)
    # log P(t|C), add-one smoothed over the feature vocabulary
    nb_ptc = np.log(1.0 + prod) - np.log(len(LDfeats) + prod.sum(0))

    # output the model (npz+LZMA, the format the runtime ships and loads)
    model = nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output
    npz_path = os.path.join(model_dir, 'model.npz.xz')
    save_model(npz_path, model)
    print(f"wrote model to {npz_path} ({os.path.getsize(npz_path)} bytes)")


if __name__ == "__main__":
    main()
