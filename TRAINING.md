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

139 languages + zxx + the two internal script-split classes (`srl`, `uzc`)
= **142 NB classes / 140 public labels** (nb merged into no 2026-08-26,
both were Bokmål), **7 domains**, 130,914 docs. Uniform
`--max-docs-per-lang 300`, docs truncated to 3,000 bytes at tokenization
(`--doc_cap`, equalizes byte weight across domains).

| Domain      | Source                    | Docs   | Register       |
|-------------|---------------------------|--------|----------------|
| wiki        | cirrus dumps              | 37,901 | encyclopedia   |
| leipzig     | Leipzig Corpora (news)    | 30,538 | news           |
| cc100       | CommonCrawl 2018 filtered | 29,006 | web            |
| tatoeba     | user sentences (CC BY)    | 18,864 | conversational |
| glotcc      | GlotCC-V1 (topup)         |  9,381 | web            |
| glot500     | Glot500 (topup)           |  4,624 | mixed          |
| glotsparse  | GlotSparse (topup)        |    600 | web            |

The last three are `topup.py` sources for thin classes, not general domains.

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
python -m py3langid.train.train -m model_dir corpus
```

Bare defaults reproduce the release config (order 2-5, doc_cap 3000,
df_tokens 60000, feats_per_lang 1050); every knob is overridable for sweeps.
Class priors are `log(per-class doc counts)`: the old `--prior_cap 1200` was
measured to be a no-op (counts run 135..1318, clipping four classes by
≤0.09 nats) and is gone, while dropping the priors altogether costs
CommonLID −519 labels (p=2e-61), so they stay as-is. Tokenization happens
once into per-(domain,lang) shards cached at `CORPUS_DIR.shards`
(`--shards` overrides), holding document frequency for every *selectable*
term: orders 2..max_order in full, plus — since byte order 5 cannot span
two 3-byte codepoints — only the CJK codepoint bigrams at order 6. A
higher-order cache serves lower orders, and `--doc_cap` is part of both the
shard filename and the cache key. All later stages are dict/numpy algebra
over shards. Deterministic: same corpus + settings → byte-identical model.
Warm run ~65s at the release config.

Three memory/size fixes landed 2026-08-31, all verified to leave the
selected feature list and the shipped model byte-identical:

- **order 6 is CJK-only.** Every other order-6 n-gram was counted and then
  dropped unread — 64% of the global term tally, 49% of shard bytes, for
  21k terms actually used. Shard cache 1.8 GB → 924 MB, global terms 41.6M
  → 15.4M. Shards written before the change record max_order 6 and stay
  valid as supersets, so clear the cache to realise the savings.
- **`merge_docfreq` chunks by a fixed 8 shards**, not one chunk per job: a
  per-job chunk grew a near-global Counter in every worker before pickling
  it back (8.7 GB parent peak at `-j 4`). `count_matrices` keeps
  one-chunk-per-job, whose partials are dense matrices.
- **count matrices are int32**, not int64: document frequencies bounded by
  docs per class (measured max 1318), and every worker allocates the full
  `nf × nlang` matrix and ships it through the pool.

Peak is now 2.4 GB at `-j 10` (merge), cold shard build 29s, merge 28s.

## Cluster features (unified 2026-08-29)

`CLUSTERS`/`CLUSTER_K` in `common.py`: each confusable cluster gets the top
`CLUSTER_K`=150 features by cluster-restricted IG that the per-language
quota missed, junk-filtered (digit/punct-only candidates skipped). Clusters:
{ms,id}, {bs,hr}, {no,nn,da}, {zh,yue,wuu}.

This is one mechanism where there used to be two. Group features and CJK
codepoint bigrams differed only in candidate pool, so the pool is now shared
(byte n-grams plus the order-6 CJK bigrams) and each cluster picks whatever
discriminates its own languages — the CJK cluster takes ~86 of its 150 slots
as bigrams and spends the rest on byte n-grams. Accuracy vs the two
special-cased subsystems is a wash: FLORES +31 (p=1.9e-03), WiLI −12
(p=0.043), OpenLID ±0, CommonLID −16 (n.s.).

Measured, do not redo:
- **K=150 is the optimum**, still. K=75 loses on both free benches
  (−42/−43). K=300 wins them (OpenLID +57, p=0.011, hr +73) but **fails the
  CommonLID gate** (−128, p=1.1e-03) via an ms/id trade (ms −50, id +49).
  Note the pre-longest-match ledger rejected K=300 for the bs/hr seesaw;
  the seesaw now nets positive, so the constant survives for a new reason.
- **Cluster members must be mutually confusable within one script.**
  Widening {bs,hr} to {bs,hr,sr,mk} costs WiLI −105 / OpenLID −82 (hr
  −48/−69): with Cyrillic members in the cluster, script-level features
  satisfy the restricted IG trivially and the budget stops buying the
  bs/hr lexical cues.
- **Arabic {ar,arz,ary} rejected again** (WiLI −28, OpenLID −24, ary −27).
  The old note blamed the gated blend covering it; the blend is gone and
  Arabic still fails, so the cause is register, not redundancy.
- **{xh,zu} rejected** (no gain: WiLI −19, OpenLID −6 n.s.), and every
  retired blend cluster at once is far worse (−115/−119).

## Longest-match emission (adopted 2026-08-28, replaced the gated blend)

The Aho-Corasick scanner emits, at each byte position, only the **longest**
matching feature instead of all of them (`out_feat` in the model: one
feature index per DFA state, -1 = none). Every feature stays in the model —
each is the longest match at its own trie node — but a position no longer
votes ~2.3 times with strictly nested n-grams, which Naive Bayes was
multiplying as if independent.

`train.py` therefore estimates the NB numerators from a **scanner corpus
pass** (`feature_counts`) rather than from shard
n-gram totals, so training counts exactly what the runtime counts.

This subsumed the gated blend, which is gone (`blend_ptc`, `BLEND_CLUSTERS`,
`build_blend`, `--no_blend`). Measured against the previous
all-matches-plus-blend model, paired McNemar, all four benchmarks positive:
WiLI 95.4367 to 95.5542 (+141, p=6e-09), OpenLID 94.3283 to 94.4596 (+315,
p=5e-10), FLORES 96.2434 to 96.3203 (+187, p=3e-07), CommonLID micro
91.7426 to 92.1755 (+1610, p=2e-85) and macro 89.969 to 90.208. Model
10.27 to 5.08 MB, classify 90.9 to 47.6 µs, RSS +237 to +136 MB. Training
cost is unchanged: the scanner pass replaces the blend's state-visit pass.

Note: with no blend, featureless input (no selected n-gram anywhere, e.g.
a 2-byte string) returns a flat score of 0.0 — a uniform distribution
under `norm_probs`, so `min_confidence` abstains. Blend arrays are ignored
if a model still ships them, but the older multi-match CSR output is not
readable: `modelio` requires `nextmove_row` and `out_feat`.

`build_scanner` returns the DFA as the file and the runtime both use it:
the distinct transition rows plus a state -> row index. A state with no
outgoing edges keeps its fail state's row, which is every duplicate there
is (38,270 rows for 104,518 states), so the flat table is never allocated
and `save_model`/`load_model` are inverses.

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
