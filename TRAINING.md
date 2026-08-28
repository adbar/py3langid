# Training a New Model

Train a langid.py model on this laptop (11 cores, 36GB RAM).
Close to the original Lui & Baldwin (2011) setting: multi-domain corpus,
LD feature selection, Multinomial NB over byte n-grams.

## Original setting (reference)

The 2011 paper used 5 domains × ~20K docs each = ~100K docs total, 97 languages.

| Domain       | Source      | Docs   | Langs | Avg doc |
|--------------|-------------|--------|-------|---------|
| legal        | JRC-Acquis  | 20,000 | 22    | 18 KB   |
| web          | ClueWeb09   | 20,000 | 10    | 37 KB   |
| encyclopedia | Wikipedia   | 20,000 | 68    | 7.5 KB  |
| news         | Reuters RCV2| 20,000 | 13    | 3.4 KB  |
| software     | Debian i18n | 21,735 | 89    | 12 KB   |

LD feature selection prefers ≥2 domains per language; more domains = better
generalization.

## Our setting

137 languages + zxx (138 classes; nb merged into no 2026-08-26, both
were Bokmål), **4 domains**,
uniform `--max-docs-per-lang 300`, docs truncated to 3,000 bytes at
tokenization (`--doc_cap`, equalizes byte weight across domains).

| Domain  | Source                    | Langs | Register       |
|---------|---------------------------|-------|----------------|
| tatoeba | user sentences (CC BY)    | ~93   | conversational |
| cc100   | CommonCrawl 2018 filtered | 85    | web            |
| wiki    | cirrus dumps              | 96    | encyclopedia   |
| leipzig | Leipzig Corpora (news)    | 89    | news           |

Dropped by measurement: **bible** (negative LOO contribution) and **opensub**
(+0.08 overall, hurt every confusable pair; decision 2026-08-26). Tatoeba is
load-bearing for the 14 languages with only 3 domains. Leave-one-out
ablations: wiki +2.35, cc100 +0.15, leipzig +0.12, tatoeba +0.03 (but
benchmarks carry no conversational register, so tatoeba is undervalued).

### Serbian: two internal script classes

Serbian is written in both scripts, and one NB class spanning two scripts
dilutes its weights (a single mixed class scores 0.1% on Cyrillic test data).
Gather-time transliteration (used until 2026-08-26) fixed Cyrillic sr but
forced all Latin-script Serbian into bs/hr — the flaw OpenLID-v3 diagnosed on
web data. Current solution: majority-Latin sr docs go into an internal `srl`
class; after training, `srl` is relabeled to `sr` in the model's class list
(duplicate labels are fine — classification returns `nb_classes[argmax]`).
Both scripts now identify as sr.

### Data quality: self-verify filter

All bespoke marker-word filters (`croatian_leaning`, `drop_english_paragraphs`,
`OPENSUB_SKIP`, per-class thickening) were removed 2026-08-26 in favor of one
general mechanism: classify every training doc with a bootstrap model and drop
docs whose prediction is a *different, non-confusable* language. Confusable
pairs (bs/hr/sr, ms/id, no/nn/da, xh/zu, hi/ne/mr, fa/ar/ps, **sr/mk**) are
protected — dropping there would launder the pair boundary through the model's
own bias. Paragraph-level verification was tested: it removes real junk
(e.g. Latin botany articles in az wiki) but cannot see contamination the
bootstrap model already learned (xh/zu English stubs) — circularity.

## gather_data.py

```
python -m py3langid.train.gather_data \
    --output corpus \
    --langs af,am,...,zu           # default: model languages
    --domains tatoeba,cc100,wiki,leipzig  # + topup (thin classes; topup.py)
    --max-docs-per-lang 300
    --sentences-per-doc 50
    --jobs 4
```

Output: `corpus/{domain}/{lang}/docNNNN.txt` (UTF-8 bytes; pipeline reads
binary, so encoding only needs to be consistent).

## Training run

```
python -m py3langid.train.train -m model_dir --prior_cap 1200 corpus
```

Defaults reproduce the release config (order 2-5, doc_cap 3000, df_tokens
30000, feats_per_lang 700); every knob is overridable for sweeps. Tokenization
happens once into per-(domain,lang) n-gram count shards cached at
`CORPUS_DIR.shards` (`--shards` overrides). Shards store orders 1..max_order
requested; a higher-order cache serves all lower orders, and `--doc_cap` is
part of both the shard filename and the cache key. All later stages are
dict/numpy algebra over shards. Deterministic: same corpus + settings →
byte-identical model. Warm run ~60s at the release config.

## Group features + gated blend (adopted 2026-08-28)

`train.py` adds two stages on top of the base LD selection (skip with
`--no_blend`):

- **Group features** (`PAIR_GROUPS`/`PAIR_K` in `common.py`): per language
  group ({ms,id}, {bs,hr}, {no,nn,da}), the top-150 new features by
  group-restricted IG, junk-filtered (digit/punct-only candidates skipped).
  K=150 measured optimal: K=300+ reintroduces bs/hr seesaw, K=75 undershoots.
  Arabic groups measured harmful (register mismatch) — handled by the blend
  cluster instead.
- **Gated blend**: a corpus pass counts per-(DFA state, lang) visits; the
  model ships `blend_ptc` (f16) = log-mixture of the state-level NB
  (α=10) and the folded feature model (λ=0.5), plus per-class dialect
  cluster ids (`BLEND_CLUSTERS`). At runtime (`langid.py`), when the main
  model's top1−top2 margin per byte < `BLEND_TAU`=0.075 (~4-12% of docs),
  the blend re-decides; a winner inside a dialect cluster is re-decided
  within the cluster by the main model. Verified (exact fold, McNemar,
  FLORES-dev held out): WiLI +0.09 / OpenLID +0.15 / FLORES +0.05 /
  CommonLID +0.52 / FLORES-dev +0.06 vs the same model without either
  stage. Costs: model 4.5→7.5 MB, classify ~44→50 µs/call, a curated
  cluster list (a new confusable class added OUTSIDE its cluster gets
  absorbed by the blend — extend `BLEND_CLUSTERS` when adding dialects).
  Blend is disabled under `set_languages()` restriction.

## Evaluation

- **Dev benchmarks** (every knob is tuned on these): `benchmarks/bench_wili.py`
  (WiLI-2018, 94K) and `benchmarks/bench_openlid.py` (OpenLID sample, 180K).
  Baseline JSONs checked in; runs print per-language deltas.
- **FLORES-200 devtest** (`benchmarks/bench_flores.py`, 92K): FROZEN held-out.
  Run once on the final candidate only. Shipped model (armE, 2026-08-27): 96.01%.
- Caveats: both dev sets are formal register (conversational domains are
  invisible); the WiLI X→en tail (~700 errors) is test contamination; bs/hr
  text is often genuinely multi-valid, so pair accuracy has an intrinsic
  ceiling (report pair sums and balance, not just accuracy).

## Results (2026-08-26; full log in model_out/sweep_results.tsv and sweep25_results.tsv)

Shipped model: WiLI 91.21 / OpenLID 88.41.

- **v4** (5 domains incl. opensub, filters, sr translit, fpl=500):
  WiLI 95.22 / OpenLID 94.92, 23,429 feats, 2.38MB.
- **Simplified 4-domain candidate** (`simpl4dom_sr2`): WiLI 95.42 /
  OpenLID 94.64, sr 95.8 with both scripts, bs+hr balanced (60/51 vs v4's
  70/38). Known cost: xh+zu −4 (English stubs back in those wikis).
- Sweep findings: `min_order=2` is a free win (smaller, 13% faster, no
  accuracy loss); fpl and df_tokens interact (df_tokens=15K binds above
  fpl≈700 — best dev cell `o5_fpl1000_dft30k`: 95.45/95.33 at 4.43MB and
  −56% throughput); order 5/6 pays only at fpl≥1000.
- **Adopted**: `tf_log1p` (inference-side sublinear TF, +0.07/+0.08 free —
  needs log1p in the runtime `_score`), 4 domains, simplified gather.
- Measured and rejected: Lidstone α≠1, order 3, uniform priors, CNB/weight
  normalization (l1norm flips bs/hr entirely — boundary is weight-scale),
  paragraph self-verify for xh/zu (circular).

## Open

- Final config choice (size/speed budget vs the fpl/df_tokens frontier),
  then FLORES gate, full re-gather, baseline re-snapshot.
- bs/hr fine calibration (per-class score scale) and Ridge/hashed-feature
  weight-swap experiments (training-only; runtime unchanged).
- Debian i18n as a possible 5th domain (software register, thickens the 14
  thin languages).
