# Training a Model

The pipeline follows Lui & Baldwin (2011): a multi-domain corpus, LD
feature selection (per-language information gain minus domain information
gain), and Multinomial Naive Bayes over byte n-grams. Training is
deterministic — the same corpus and settings reproduce the model byte for
byte. Timings below are from an 11-core, 36 GB laptop.

Training requires only `numpy` (already a dependency of py3langid). Data
gathering additionally needs `huggingface_hub` and `datasets` for the
top-up sources: `pip install huggingface_hub datasets`.

## Corpus

Layout: `corpus/{domain}/{lang}/docNNNN.txt`, UTF-8 bytes. A domain is a
text register (news, web, encyclopedia, ...); LD selection needs ≥2 domains
per language to separate language signal from domain signal, so more
domains generalize better.

The released model is trained on 130,914 docs, capped at 300 docs per
language per domain and 3,000 bytes per doc (`DOC_CAP` in `common.py`, the
pipeline's one doc byte budget — gathering, tokenization, the verifier and
`zxx` all read it):

| Domain      | Source                    | Docs   | Register       |
|-------------|---------------------------|--------|----------------|
| wiki        | cirrus dumps              | 37,901 | encyclopedia   |
| leipzig     | Leipzig Corpora (news)    | 30,538 | news           |
| cc100       | CommonCrawl 2018 filtered | 29,006 | web            |
| tatoeba     | user sentences (CC BY)    | 18,864 | conversational |
| glotcc      | GlotCC-V1 (topup)         |  9,381 | web            |
| glot500     | Glot500 (topup)           |  4,624 | mixed          |
| glotsparse  | GlotSparse (topup)        |    600 | web            |

The last three are top-up sources: `topup.py` fills classes that fall
below 600 docs or 2 domains, they are not gathered for every language.

Classes: 139 languages + `zxx` (synthetic not-a-language) + two internal
script-split classes = 142 NB classes / 140 public labels. A language
written in two scripts trains as two classes and is merged back to one
label at model assembly (`sr` Cyrillic + internal `srl` Latin; same for
`uz`/`uzc`) — a single class spanning two scripts dilutes its weights.
Adding a split language is one `SplitScript` entry in `common.py`.

## Gathering data

```
python -m py3langid.train.gather_data \
    --output corpus \
    --langs af,am,...,zu                  # default: model languages
    --domains tatoeba,cc100,wiki,leipzig  # 'topup' fills thin classes
    --max-docs-per-lang 300
    --sentences-per-doc 50
    --jobs 4
```

Downloads are cached in `raw_downloads/` and re-gathers resume where they
stopped.

## Corpus hygiene

On a fresh corpus, in order:

```
python -m py3langid.train.zxx corpus     # generate the not-a-language class
python -m py3langid.train.dedup corpus   # cross-domain exact line dedup
python -m py3langid.train.verify --model VERIFIER_MODEL corpus
```

`verify` classifies every doc and moves docs predicted as a *different,
non-confusable* language to a sibling `corpus_dropped/` tree (nothing is
deleted). Confusable pairs (bs/hr/sr, ms/id, no/nn/da, ...) are protected:
dropping there would push the pair boundary toward the verifier's own
bias. Use an independent verifier for the first round (e.g. the previous
release model or any model not trained on this corpus) — a model trained
on the corpus itself accepts the contamination it learned. After one clean
round, the retrained model is a valid self-verifier. `--paragraphs`
strips foreign paragraphs in place instead of dropping docs.

## Training run

```
python -m py3langid.train.train -m model_dir corpus
```

Bare defaults reproduce the release config, byte for byte. Warm run ~57s
at `-j 10`, cold ~29s more for tokenization.

`--feats_per_lang` is the only sweep knob as a flag. Tokenization
constants live in `common.py` (`MIN_NGRAM_ORDER`, `MAX_NGRAM_ORDER`,
`SELECT_ORDERS`, `DF_TOKENS`, `DOC_CAP`) — sweep by editing; the shard
cache invalidates itself.

How it works:

- Tokenization writes per-(domain, lang) document-frequency shards cached
  at `CORPUS_DIR.shards` (`--shards` overrides). Only changed directories
  are re-tokenized; all later stages are algebra over the shards.
- Order 6 is restricted to CJK codepoint bigrams (byte order 5 cannot
  span two 3-byte codepoints).
- Feature selection: top `DF_TOKENS` terms per order as candidates, then
  top `feats_per_lang` per language by LD weight, restricted to terms
  appearing in ≥2 domains.
- Confusable clusters (`CLUSTERS` in `common.py`) each add 150 features
  by cluster-restricted IG. Change them only with a full re-evaluation.
- Class priors are `log(per-class doc counts)`, no smoothing knobs.

## Longest-match emission

The compiled Aho-Corasick scanner emits only the **longest** matching
feature at each byte position (`out_feat` in the model), avoiding
duplicate votes from nested n-grams. NB numerators are counted by a
scanner pass over the corpus (`feature_counts`) that runs the runtime's
own walk, so training counts exactly what classification accumulates.

## Evaluation

- Dev (tuned on): `bench_wili.py` (WiLI-2018) and `bench_openlid.py`
  (OpenLID). Checked-in baseline JSONs track the shipped model; runs
  print per-language deltas, `--save-baseline` re-snapshots.
- Held-out (run once per adoption): `bench_flores.py` (FLORES-200) and
  `bench_commonlid.py` (CommonLID, noisy-register web text — decides ties).
- Caveats: dev sets are formal register; some confusable pairs (bs/hr
  especially) are genuinely multi-valid with an intrinsic accuracy ceiling.

## Results

Shipped model: **WiLI 95.51 / OpenLID 94.55**, 140 labels, 100,053
features, 4.6 MB. The pre-fork langid.py model scores 91.21/88.41 on the
same harnesses. Sweep logs: `model_out/sweep_results.tsv` and
`model_out/sweep25_results.tsv`.

## Quick recipe

End-to-end from scratch (assuming the previous release model as verifier):

```bash
# 1. gather
python -m py3langid.train.gather_data --output corpus --jobs 4
# 2. hygiene
python -m py3langid.train.zxx corpus
python -m py3langid.train.dedup corpus
python -m py3langid.train.verify --model old_model/model.npz.xz corpus
# 3. train
python -m py3langid.train.train -m model_out corpus
# 4. evaluate
python benchmarks/bench_wili.py --model model_out/model.npz.xz
python benchmarks/bench_openlid.py --model model_out/model.npz.xz
```
