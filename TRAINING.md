# Training a Model

Naive Bayes over byte n-grams with LD feature selection (Lui & Baldwin 2011),
plus a word table for short input. Training is deterministic.

Run from a clone, since `py3langid.train` is not in the wheel. It needs
`numpy`, and the top-up sources also need `huggingface_hub` and `datasets`.

```bash
git clone https://github.com/adbar/py3langid.git && cd py3langid
python -m py3langid.train.gather_data --output corpus --domains tatoeba,cc100,wiki,leipzig,topup
python -m py3langid.train.train -m model_dir corpus
```

## Corpus

Layout: `corpus/{domain}/{lang}/docNNNN.txt`. LD selection needs at least two
domains per language. Docs are capped at 3,000 bytes (`DOC_CAP`).

| Domain  | Source                  | Docs    |
|---------|-------------------------|---------|
| wiki    | Wikipedia article leads | 112,824 |
| leipzig | Leipzig news            |  29,856 |
| cc100   | CommonCrawl             |  29,503 |
| tatoeba | user sentences          |  18,745 |
| glotcc  | GlotCC-V1 (topup)       |  11,726 |
| glot500 | Glot500 (topup)         |   1,391 |

`MANIFEST.json` records the settings and raw file hashes. The release used
`--wiki-date 20260913`.

`sr`, `uz` and `zh` train as two script classes each, merged into one label
(`SPLIT_SCRIPT`). `wuu` stays one class (`SCRIPT_QUOTA`).

## Gathering and cleaning

The defaults are 300 docs per language and domain, and 900 for wiki. `topup`
fills thin classes from GlotCC, and uses Glot500 only for `GLOT500_CLASSES`.
Downloads are cached and a rerun resumes. `clean` runs within each gather and
also on its own:

- `zxx` generates the not-a-language class.
- `dedup` removes duplicate lines per language, and lines found in the
  eval sets (hashes in `train/eval_lines.npy`, regenerate when sources change).
- `rules` fixes sources that carry another language (`hbo`, `arz`, `yue`,
  `gom`, `xh`) and drops docs left under 500 bytes.

## Training

`--feats_per_lang` (default 1600) is the only tuning flag. Other constants
live in `common.py` and `words.py`. A run with a warm shard cache takes
about a minute.

- Shards cache n-gram (orders 2 to 5), word and CJK document frequencies
  per directory, plus Han character and pair counts.
- Selection keeps the top `DF_TOKENS` terms per order, then the top
  `feats_per_lang` per language by LD weight. The weight is discounted by
  the share of languages that also use the feature, since each one stores
  a count.
- NB counts come from the runtime's own longest-match walk. Counts up to
  `COUNT_FLOOR` are zeroed. Features left without counts are dropped and the
  rest recounted: longer matches shadow them.
- The word table adds PMI credits from document frequencies, so short input
  separates close languages (ms/id, bs/hr). Long words need more documents. Tokens keep combining marks, so Indic words stay
  whole. CJK marker characters and Han character pairs get a fixed credit.
  Han pair bigram credits separate Mandarin, Cantonese and Wu.

## Results

Dev sets are WiLI-2018 and OpenLID. FLORES+ and CommonLID are held out.
FLORES+ is pinned at v4.5 and used for evaluation only. It fixes the
FLORES-200 Cantonese set, which was standard Chinese (97.44 on FLORES-200).
The harness is not in the repository.

| | WiLI | OpenLID | FLORES+ | CommonLID |
|---|---|---|---|---|
| Shipped model | 96.13 | 96.46 | 98.27 | 95.02 |
| 2-word prefixes | 76.27 | 85.37 | | |
| Pre-fork langid.py | 91.21 | 88.41 | | |

The model has 139 labels and 116,519 features, and weighs 6.5 MB.
