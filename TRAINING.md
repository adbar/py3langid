# Training a Model

The pipeline follows Lui & Baldwin (2011): a multi-domain corpus, LD
feature selection (per-language information gain minus domain information
gain), and Multinomial Naive Bayes over byte n-grams. Training is
deterministic — the same corpus and settings reproduce the model byte for
byte.

The `py3langid.train` package ships in the repository, not in the PyPI
wheel, so run the commands below from a clone:

```bash
git clone https://github.com/adbar/py3langid.git
cd py3langid
```

Training itself requires only `numpy` (already a dependency of py3langid).
Data gathering additionally needs `huggingface_hub` and `datasets` for the
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
below 600 docs or 2 domains from GlotCC, Glot500, GlotSparse and, as a last
resort, UDHR; they are not gathered for every language. Two classes (`sdh`,
`uzs`) exist only in GlotSparse.

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
    --domains tatoeba,cc100,wiki,leipzig  # add 'topup' to fill thin classes
    --max-docs-per-lang 300
    --sentences-per-doc 50
    --jobs 4
```

`topup` is not in the default domain list, so pass `--domains` explicitly
for a corpus that covers every shipped label. Downloads are cached in
`raw_downloads/` and re-gathers resume where they stopped. Run the stages
below in order on the result, then evaluate before adopting the model.

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

Bare defaults reproduce the release config, byte for byte. A run with a
warm shard cache takes about a minute at `-j 10`; a cold one adds the
tokenization pass.

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
- The compiled Aho-Corasick scanner emits only the **longest** matching
  feature at each byte position (`out_feat` in the model), avoiding
  duplicate votes from nested n-grams. NB numerators come from a scanner
  pass over the corpus (`feature_counts`) that runs the runtime's own walk,
  so training counts exactly what classification accumulates.

## Evaluation

The eval harness is not part of the repository; the datasets are large and
licensed separately. The split used for this model:

- Dev (tuned on): WiLI-2018 and OpenLID, scored against a stored baseline
  per language so a run reports deltas rather than absolute numbers.
- Held-out (run once per adoption): FLORES-200 and CommonLID
  (noisy-register web text — decides ties).
- Caveats: dev sets are formal register; some confusable pairs (bs/hr
  especially) are genuinely multi-valid with an intrinsic accuracy ceiling.

## Results

Shipped model: **WiLI 95.51 / OpenLID 94.55**, 140 labels, 100,053
features, 4.6 MB. The pre-fork langid.py model scores 91.21/88.41 on the
same harnesses.
