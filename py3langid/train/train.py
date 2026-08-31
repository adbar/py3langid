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
    CJK_DF_FLOOR,
    CLUSTER_K,
    CLUSTERS,
    DF_TOKENS,
    DOC_CAP,
    FEATURES_PER_LANG,
    LABEL_ALIAS,
    MAX_NGRAM_ORDER,
    MIN_NGRAM_ORDER,
    TOKENIZE_ORDER,
)
from .scanner import build_scanner
from .shards import build_shards, count_matrices, domain_presence, merge_docfreq
from .stages import (
    cluster_features,
    compute_IG,
    compute_IG_binarized,
    feature_counts,
    index_corpus,
    is_cjk_bigram,
    learn_pc,
    ngram_select,
    prod_to_ptc,
    select_LD_features,
    state_visit_counts,
)


def _axis(names, values):
    """One class axis (languages or domains), in `names` order.
    @returns (doc counts, per-name counts as an array, name -> column)"""
    counts = Counter(values)
    return (counts,
            np.array([counts[name] for name in names]),
            {name: i for i, name in enumerate(names)})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model", help="save output to MODEL_DIR", metavar="MODEL_DIR")
    parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no parallelization)")
    parser.add_argument("--max_order", type=int, help="highest n-gram order to use", default=MAX_NGRAM_ORDER)
    parser.add_argument("--min_order", type=int, help="lowest n-gram order to use", default=MIN_NGRAM_ORDER)
    parser.add_argument("--doc_cap", type=int, default=DOC_CAP,
        help="truncate each doc to N bytes at tokenization (0 = no cap)")
    parser.add_argument("--df_tokens", type=int, help="candidate pool: top tokens by document frequency per order", default=DF_TOKENS)
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

    # index the corpus; sorted walk order defines the class column order
    items, langs, domains = index_corpus(args.corpus)
    lang_counts, lang_dist, lang_index = _axis(langs, (lang for _, lang, _ in items))
    domain_counts, domain_dist, domain_index = _axis(
        domains, (domain for domain, _, _ in items))

    print(f"langs({len(langs)}): " + ' '.join(f"{lang}({lang_counts[lang]})" for lang in langs))
    print(f"domains({len(domains)}): " + ' '.join(f"{d}({domain_counts[d]})" for d in domains))
    print(f"identified {len(items)} files")

    # Tokenize the corpus into per-(domain, lang) n-gram count shards.
    # Cached shards are reused; only changed directories are re-tokenized.
    shard_dir = args.shards or os.path.normpath(args.corpus) + '.shards'
    shard_items = build_shards(items, shard_dir,
                               max(TOKENIZE_ORDER, args.max_order),
                               args.jobs, args.doc_cap)

    doc_count = merge_docfreq(shard_items, args.jobs)
    print(f"tallied document frequency of {len(doc_count)} terms")

    DFfeats = ngram_select(doc_count, args.max_order, args.df_tokens, args.min_order)
    df_set = set(DFfeats)
    cjk_cand = sorted(t for t, c in doc_count.items()
                      if c >= CJK_DF_FLOOR and t not in df_set and is_cjk_bigram(t))
    doc_count = None
    features = DFfeats + cjk_cand
    n_df = len(DFfeats)
    print(f"selected {n_df} DF features + {len(cjk_cand)} CJK candidates")

    # Compute IG (and the NB numerators, in the same shard pass)
    # prod (shard n-gram totals) is not retained: the NB numerators are
    # counted below under the scanner's longest-match emission instead
    cm_lang, cm_domain, _ = count_matrices(shard_items, features, lang_index,
                                           domain_index, args.jobs)

    # Select features by LD weight (per-language IG minus domain IG),
    # candidates restricted to terms seen in >=2 of the language's domains
    print("computing information gain")
    ld = (compute_IG_binarized(cm_lang[:n_df], lang_dist)
          - compute_IG(cm_domain[:n_df], domain_dist)[:, None])
    domcount, lang_domains = domain_presence(shard_items, DFfeats, lang_index)
    need = np.array([min(2, lang_domains[lang]) for lang in langs])
    present = domcount >= need[None, :]
    LDidx = select_LD_features(ld, args.feats_per_lang, present)
    extra = cluster_features(cm_lang, lang_dist, lang_index, features, LDidx,
                             CLUSTERS, CLUSTER_K)
    n_cjk = sum(i >= n_df for i in extra)
    print(f"added {len(extra)} cluster features ({n_cjk} CJK bigrams)")
    LDidx |= extra
    # one order for both LDfeats and nb_ptc's rows
    LDorder = sorted(LDidx, key=features.__getitem__)
    LDfeats = [features[i] for i in LDorder]
    print(f'selected {len(LDfeats)} features')

    # Compile a scanner for the LDfeats (one longest match per position)
    tk_nextmove, tk_output = build_scanner(LDfeats)
    emitting = sum(f >= 0 for f in tk_output)
    print(f"scanner: {len(tk_output)} states, {emitting} emitting")

    # Assemble the NB model (duplicate labels are fine:
    # classification returns nb_classes[argmax])
    nb_classes = [LABEL_ALIAS.get(lang, lang) for lang in langs]
    # log P(class) from the per-class doc counts. The old --prior_cap 1200
    # was measured to be a no-op (counts run 135..1318, so it clipped four
    # classes by <=0.09 nats) and is gone: bare defaults now reproduce the
    # release. The priors themselves DO earn their place -- dropping them
    # costs CommonLID -519 labels (p=2e-61).
    nb_pc = learn_pc(lang_dist)

    # NB numerators: count what the runtime accumulates, i.e. one longest
    # match per byte position, rather than every n-gram occurrence
    print("counting longest-match feature occurrences")
    state_counts = state_visit_counts(items, tk_nextmove, len(tk_output),
                                      lang_index, args.doc_cap)
    nb_ptc = prod_to_ptc(feature_counts(state_counts, tk_output, len(LDfeats)))

    # output the model (npz+LZMA, the format the runtime ships and loads)
    model = nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output
    npz_path = os.path.join(model_dir, 'model.npz.xz')
    save_model(npz_path, model)
    print(f"wrote model to {npz_path} ({os.path.getsize(npz_path)} bytes)")


if __name__ == "__main__":
    main()
