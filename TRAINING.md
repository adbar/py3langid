# Training a Model

The pipeline follows Lui & Baldwin (2011): a multi-domain corpus, LD
feature selection (per-language information gain minus domain information
gain), and Multinomial Naive Bayes over byte n-grams. Training is
deterministic — the same corpus and settings reproduce the model byte for
byte. Timings below are from an 11-core, 36 GB laptop.

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
bias. Use an independent verifier for the first round — a model trained on
the corpus itself accepts the contamination it learned. After one clean
round, the retrained model is a valid self-verifier. `--paragraphs`
strips foreign paragraphs in place instead of dropping docs.

## Training run

```
python -m py3langid.train.train -m model_dir corpus
```

Bare defaults reproduce the release config, byte for byte.
`--feats_per_lang` is the only sweep knob left as a flag: everything that
decides what tokenization writes into the cache is a constant in
`common.py` (`MIN_NGRAM_ORDER`, `MAX_NGRAM_ORDER`, `SELECT_ORDERS`,
`DF_TOKENS`, `DOC_CAP`), since a flag there is a cache-key axis. Sweep by
editing the constant; the cache invalidates itself. Warm run ~57s at
`-j 10`, cold ~29s more for tokenization.

How it works:

- Tokenization runs once into per-(domain, lang) document-frequency
  shards cached at `CORPUS_DIR.shards` (`--shards` overrides). Only
  changed directories are re-tokenized; all later stages are algebra over
  the shards. A shard's cache key covers its documents *and* the
  tokenization constants, and reuse needs exact equality, so editing an
  order constant re-tokenizes rather than serving stale shards.
- Order 6 is restricted to CJK codepoint bigrams: byte order 5 cannot span
  two 3-byte codepoints, so those are the only order-6 terms worth having.
  `doc_ngrams` enforces this alone — the orders are in the shard cache key,
  so a shard written under other rules is never read.
- Feature selection takes the top `DF_TOKENS` terms per order as
  candidates (order 6 = the CJK bigrams), then the top `feats_per_lang`
  per language by LD weight, restricted to terms the language shows in
  ≥2 domains.
- Confusable clusters ({ms,id}, {bs,hr}, {no,nn,da}, {zh,yue,wuu};
  `CLUSTERS` in `common.py`) each add the 150 best features by
  cluster-restricted information gain that the per-language quotas
  missed. The cluster list and budget were swept extensively — change
  them only with a full re-evaluation.
- Class priors are `log(per-class doc counts)`; there are no smoothing
  knobs.

## Longest-match emission

The compiled Aho-Corasick scanner emits, at each byte position, only the
**longest** matching feature instead of all of them (`out_feat` in the
model: one feature index per DFA state, -1 = none). Every feature stays in
the model — each is the longest match at its own trie node — but a
position no longer votes several times with strictly nested n-grams,
which Naive Bayes would multiply as if independent.

The NB numerators are therefore counted by a scanner pass over the corpus
(`feature_counts`) that runs the runtime's own walk, so training counts
exactly what classification accumulates. Featureless input (no selected
n-gram anywhere) scores as a uniform distribution under `norm_probs`, so
`min_confidence` abstains on it.

## Evaluation

- Dev benchmarks (every knob is tuned on these): `benchmarks/bench_wili.py`
  (WiLI-2018, 120K mapped samples) and `benchmarks/bench_openlid.py`
  (OpenLID sample, 240K). Checked-in baseline JSONs track the shipped
  model; runs print per-language deltas against them, and
  `--save-baseline` re-snapshots after adopting a new model.
- Held-out gates, run once per adoption and never tuned on:
  `benchmarks/bench_flores.py` (FLORES-200 dev+devtest, 243K; no training
  source uses FLORES) and `benchmarks/bench_commonlid.py` (CommonLID,
  real web text with native-speaker labels — the only noisy-register set,
  so it decides ties).
- Caveats: the dev sets are formal register, so conversational gains are
  invisible; some confusable pairs (bs/hr especially) are genuinely
  multi-valid, so their accuracy has an intrinsic ceiling.

## Results

Shipped model: **WiLI 95.51 / OpenLID 94.55**, 140 labels, 100,053
features, 4.6 MB. The pre-fork langid.py model scores 91.21/88.41 on the
same harnesses. Sweep logs: `model_out/sweep_results.tsv` and
`model_out/sweep25_results.tsv`.
