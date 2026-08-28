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
    CJK_CLUSTER,
    CJK_DF_FLOOR,
    CJK_K,
    DOC_CAP,
    FEATURES_PER_LANG,
    LABEL_ALIAS,
    MAX_NGRAM_ORDER,
    MIN_NGRAM_ORDER,
    NEEDY_DF_TOKENS,
    PAIR_GROUPS,
    PAIR_K,
    TOKENIZE_ORDER,
    TOP_DOC_FREQ,
)
from .scanner import build_scanner
from .shards import build_shards, count_matrices, domain_presence, merge_docfreq
from .stages import (
    build_blend,
    compute_IG,
    compute_IG_binarized,
    group_features,
    index_corpus,
    is_cjk_bigram,
    learn_pc,
    ngram_select,
    prod_to_ptc,
    select_quota_features,
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
    parser.add_argument("--df_tokens", type=int, help="number of tokens to consider for each n-gram order", default=TOP_DOC_FREQ)
    parser.add_argument("--feats_per_lang", type=int, metavar='N', help="select top N features for each language", default=FEATURES_PER_LANG)
    parser.add_argument("--shards", metavar="SHARD_DIR", help="n-gram count shard cache (default: CORPUS_DIR.shards)")
    parser.add_argument("--prior_cap", type=int, default=0,
        help="clip per-class doc counts to N for the class priors only (0 = off)")
    parser.add_argument("--no_blend", action="store_true",
        help="skip group features and the gated-blend table")
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

    DFfeats = ngram_select(doc_count, args.max_order, NEEDY_DF_TOKENS, args.min_order)
    base_set = set(ngram_select(doc_count, args.max_order, args.df_tokens, args.min_order))
    df_set = set(DFfeats)
    cjk_cand = sorted(t for t, c in doc_count.items()
                      if c >= CJK_DF_FLOOR and t not in df_set and is_cjk_bigram(t))
    doc_count = None
    features = DFfeats + cjk_cand
    n_df = len(DFfeats)
    print(f"selected {n_df} DF features + {len(cjk_cand)} CJK candidates")

    # Compute IG (and the NB numerators, in the same shard pass)
    cm_lang, cm_domain, prod_df = count_matrices(shard_items, features, lang_index, domain_index, args.jobs)

    # Select features by LD weight (per-language IG minus domain IG),
    # candidates restricted to terms seen in >=2 of the language's domains
    print("computing information gain")
    ld = (compute_IG_binarized(cm_lang[:n_df], lang_dist)
          - compute_IG(cm_domain[:n_df], domain_dist)[:, None])
    domcount, lang_domains = domain_presence(shard_items, DFfeats, lang_index)
    need = np.array([min(2, lang_domains[lang]) for lang in langs])
    present = domcount >= need[None, :]
    base_mask = np.fromiter((f in base_set for f in DFfeats), dtype=bool, count=n_df)
    LDidx = select_quota_features(ld, present, base_mask, langs, args.feats_per_lang)
    if not args.no_blend:
        # group features are selected in base-pool row space
        idx_base = np.flatnonzero(base_mask)
        back = {int(g): k for k, g in enumerate(idx_base)}
        base_feats = [DFfeats[i] for i in idx_base]
        pair_idx = group_features(cm_lang[idx_base], lang_dist, lang_index, base_feats,
                                  {back[i] for i in LDidx if base_mask[i]},
                                  PAIR_GROUPS, PAIR_K)
        print(f"added {len(pair_idx)} group features")
        LDidx |= {int(idx_base[i]) for i in pair_idx}
        if cjk_cand and all(la in lang_index for la in CJK_CLUSTER):
            cols = [lang_index[la] for la in CJK_CLUSTER]
            cjk_ig = compute_IG(cm_lang[n_df:][:, cols], lang_dist[cols])
            LDidx |= {n_df + int(t) for t in np.argsort(cjk_ig)[::-1][:CJK_K]}
            print(f"added {min(CJK_K, len(cjk_cand))} CJK bigram features")
    # one order for both LDfeats and nb_ptc's rows
    LDorder = sorted(LDidx, key=features.__getitem__)
    LDfeats = [features[i] for i in LDorder]
    print(f'selected {len(LDfeats)} features')

    # Compile a scanner for the LDfeats
    tk_nextmove, tk_output = build_scanner(LDfeats)

    # Assemble the NB model (duplicate labels are fine:
    # classification returns nb_classes[argmax])
    nb_classes = [LABEL_ALIAS.get(lang, lang) for lang in langs]
    # priors only; feature estimates still use every doc
    pc_dist = lang_dist.clip(max=args.prior_cap) if args.prior_cap else lang_dist
    nb_pc = learn_pc(pc_dist)
    # NB numerators (counted with the IG matrices), rows in LDfeats order
    nb_ptc = prod_to_ptc(prod_df[LDorder])

    blend = None
    if not args.no_blend:
        print("counting state visits for the blend table")
        blend = build_blend(items, lang_index, nb_classes, nb_ptc,
                            tk_nextmove, tk_output, args.doc_cap)

    # output the model (npz+LZMA, the format the runtime ships and loads)
    model = nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output
    npz_path = os.path.join(model_dir, 'model.npz.xz')
    save_model(npz_path, model, blend)
    print(f"wrote model to {npz_path} ({os.path.getsize(npz_path)} bytes)")


if __name__ == "__main__":
    main()
