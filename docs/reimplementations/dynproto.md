# DynProto reimplementation notes

Paper: Y. Wu, X. Lu, R. Lai, Q. Chen, J.-X. Zhuang, W.-S. Zheng, R. Wang, *DynProto: Dynamic Prototype Evolution for OOD Detection*, IEEE TMM, arXiv 2604.23729v2 (10 pages, no appendix). Local copy: `papers/4_Test_time_OOD_detection_with_dynamic_memory/4.3_DynProto_arXiv2026.pdf`.

Code: `src/cure_ood/methods/testtime/dynproto.py` (registered as `dynproto`). Tests: `tests/test_dynproto.py`.

## Official code

`github.com/wyqstan/DynProto` (first author, same account as the DCAC release) exists but contains only a 73-byte `README.md` (single commit, 2026-05-17; checked 2026-10-13, no branches or forks). There is nothing to port, so this is a reimplementation from the paper. The adaptive threshold (Eq. 5) is credited by the paper to OWTTT [28], whose official code (`Yushu-Li/OWTTT`, `compute_os_variance` and `np.arange(0, 1, 0.01)` search) was used to resolve that step.

The round-53 version is kept unchanged as `dynproto_r53` (`methods/testtime/dynproto_r53.py`) so that `tests/test_golden_testtime.py` still reproduces the published round-53 numbers. Experiment configs that say `dynproto` now run the version described here.

## Paper element -> code

Line numbers refer to `dynproto.py`. "Alg. 1 l.k" = line k of Algorithm 1 (p. 4).

| Paper element | Code | Status |
|---|---|---|
| Image features `v = F(x)` (§III-D, Alg. 1 l.1) | `stream.x` (L2-normalised penultimate features) | Faithful up to normalisation; paper does not say whether `v` is normalised (CLIP features usually are). Normalised features are needed for a meaningful BIRCH radius. |
| ID prototypes `p_c = (1/N_c) Σ_i v_i^c`, one per class, from **ID training data aggregated by class** (§III-D, Alg. 1 l.12) | `fit`, l.86-90: mean of `bench.G` per **true** label `bench.train_labels` | Faithful. Old version grouped by *predicted* class. The mean is re-normalised; irrelevant for cosine. |
| Scoring Eq. 4: `S = Σ_i e^{cos(v,p_i)/τ} / (Σ_i e^{cos(v,p_i)/τ} + K Σ_j e^{cos(v,p_j)/τ})` (§III-D) | `neglabel_log_odds` l.29; output `-logit(S)` at l.128 | Faithful. We return `-logit(S) = -(lse_ID - log K - lse_OOD)`, a strictly monotone transform of `-S`, so every AUROC/FPR is identical to using `S` while avoiding float saturation of `S` at 0/1 when `τ = 0.01`. Tested against the direct formula (`test_eq4_score_matches_direct_formula`). |
| Coefficient `K = 5` (§III-D, §IV-A, Fig. 5) | `k=5.0` | Faithful. |
| Temperature `τ` (Eq. 3/4) | `temp=0.01` | **Not given in the paper.** We use NegLabel's default `τ = 0.01`, since the paper states it uses the NegLabel scoring function. |
| Negative prototypes | OOD prototypes `p_{C+1..C+M}` built from test caches; no text / WordNet negatives | Faithful: DynProto *replaces* NegLabel's text negatives by image OOD prototypes (§III-A last sentence). Nothing to drop. |
| One cache per ID class, initially empty (§III-C, Table V "Empty") | `caches`, l.99 | Faithful. |
| Sample goes to the cache of its **predicted** class `ŷ` (§III-C, Alg. 1 l.9) | l.119, `yhat = argmax(logits)` | Faithful. |
| Cache capacity `m = 30` per class, FIFO (drop oldest) (§III-C, §IV-A) | `deque(maxlen=cache_size)`, `cache_size=30` | Faithful (`test_fifo_keeps_most_recent`). |
| Cold start: first `T_cold = 5` iterations use base detector MSP (unimodal) / MCM (VLM) (§III-E, §IV-A) | `t_cold=5`, `t` = batch index; base = MSP from logits | Faithful for the unimodal branch. MCM (VLM branch) is not applicable: no text encoder. |
| Collection threshold `θ` = `β`-th percentile (β = 5) of the base score on ID **training** data (§III-E, §IV-A) | `fit`, l.92: 5th percentile of train MSP; admission `MSP < θ` at l.114 | Faithful. "Training data" = `bench.train_logits` (the gallery split; for ImageNet-200 the gallery half of the ID split). |
| After cold start: admission `S_DynProto < α` with `S` computed with the prototypes **before** this batch's update (Alg. 1 l.2-4, l.8) | l.108-112 | Faithful. |
| `t ≥ T_cold` but cache still empty: neither branch of Alg. 1 l.8 can fire | l.116: no admission; score stays MSP | Literal reading of Alg. 1; documented edge case (happens only if the cold start collected nothing). |
| Adaptive threshold `α`, Eq. 5, "as in OWTTT", search over `(0,1)`, per batch (`N_b` = batch size) (§III-E) | `adaptive_alpha`, `within_variance`, `ALPHA_GRID` l.17-53 | See ambiguity A1. Default `alpha_rule="owttt"`: OWTTT's weighted within-class variance on grid `0, 0.01, ..., 0.99`; one-sided groups are skipped; `>=` goes to the "ID" side. `alpha_rule="eq5"` implements Eq. 5 as printed (unweighted sum of the two group variances). Computed on the current batch only (OWTTT uses a score queue; the paper says per batch). |
| Fine-grained refinement: BIRCH on each non-empty cache separately; samples in one cluster averaged into one OOD prototype (§III-D, Alg. 1 l.13) | `birch_prototypes` l.56, called l.122 for classes whose cache changed | Faithful. Recomputing only changed caches is equivalent because BIRCH is deterministic and unchanged caches give the same prototypes. Prototypes are L2-normalised (no effect on cosine). |
| BIRCH hyper-parameters | `birch_threshold=0.5`, `birch_branching=50`, `n_clusters=None` | **Not given in the paper.** scikit-learn defaults for `threshold`/`branching_factor`. `n_clusters=None` (no global agglomerative step, so the number of prototypes is data-driven) because the paper contrasts BIRCH with K-means, which "requires manually specifying the number of clusters" (§IV-C, Table VI). scikit-learn's default `n_clusters=3` would contradict that. |
| Final score for batch `t` = Eq. 4 with the prototypes **after** this batch's cache update; base score only while the cache is empty (Alg. 1 l.11-17) | l.126-131 | Faithful. Old version scored with pre-update prototypes and returned a constant score while the cache was empty. |
| Batch size 512 (§IV-A) | `batch=128` | **Forced deviation**: our protocol scores every test-time method on batches of 128. Affects `α` (estimated from fewer scores) and the update frequency. |
| Five shuffling seeds, ID and OOD mixed and shuffled (§IV-A) | Benchmark streams (`Benchmark.stream`, seeds, mixing ratio π) | Protocol is ours (OpenOOD v1.5 splits, π ∈ {0.05, 0.2, 0.5}); paper uses all ID + all OOD test samples (≈1:1 on ImageNet; Table VII studies other ratios). |
| Per-dataset hyper-parameters | single setting | Paper uses one setting for every dataset (`m = 30, β = 5, T_cold = 5, K = 5`), as do we. |
| Backbones: DenseNet-101 (CIFAR-100), ResNet-50 (ImageNet), CLIP-B/16 (§IV-A) | frozen OpenOOD v1.5 checkpoints (ResNet-18 for CIFAR/ImageNet-200), DINOv2 skipped (`requires_logits`) | **Forced deviation**: features and logits are precomputed by our pipeline. |
| Gaussian-noise injection for extreme ID:OOD ratios ("DynProto+noise", §IV-C, Table VII) | not implemented | Optional analysis variant, not part of the method in Alg. 1. |
| Alternative cache inits (C-Out/T-Out/D-Out), RH update, K-means/AP/VB-GMM clustering, other base detectors (§IV-B/C) | not implemented | Ablations only; the main method uses Empty init, FIFO, BIRCH, MSP. |
| Memory trace for DCR | `Output.memory`: `(batch_start, indices in all caches after the update)` | Ours; the indices are exactly the samples whose features formed the OOD prototypes used to score that batch. |

## Unspecified points and the choice made

- **A1. Eq. 5.** As printed, both sums run over all `N_b` samples (the indicator is missing on the outer sum) and the two variances are not weighted by group size, while the text says the threshold is "computed as follows in OWTTT", whose code minimises the *weighted* within-class variance (Otsu). We default to OWTTT's code (`alpha_rule="owttt"`) because the paper defers to it, and keep the printed form (`alpha_rule="eq5"`, with the outer sum restricted to group members). On the smoke stream both give the same AUROC to 3 decimals (0.9077 vs 0.9070).
- **A2. Grid for α.** "We search α over the interval (0, 1)" without a step; we use OWTTT's `np.arange(0, 1, 0.01)` (`α = 0` is always rejected as one group is empty). If every threshold is rejected (constant batch) `argmin` returns 0 and nothing is admitted, as in OWTTT.
- **A3. Temperature τ.** Not stated; NegLabel default 0.01.
- **A4. BIRCH `threshold`, `branching_factor`, `n_clusters`.** Not stated; see table.
- **A5. Feature normalisation** of `v`, of the ID prototype average and of cluster means: not stated; L2-normalised features, cosine makes prototype scale irrelevant.
- **A6. Per-sample vs per-batch updates.** Algorithm 1 is written for "test sample x" with "batch index t", but `α` (Eq. 5) needs a batch of scores and `T_cold` counts iterations. We process a batch at a time: score/admit the whole batch, update caches in stream order, re-cluster, then score the batch.
- **A7. Which ID training statistics.** `θ` and ID prototypes use the ID data outside the stream (`bench.G`, `bench.train_labels`, `bench.train_logits`); for ImageNet-200 this is the gallery half of the class-stratified split, not the full ImageNet-1k training set.

## Parts that cannot be reproduced

- The VLM branch (CLIP-B/16 features, MCM as base detector, the CLIP numbers in Tables I-VIII) needs a text encoder; our benchmarks only provide frozen image features and classifier logits.
- The exact backbones (DenseNet-101, ResNet-50 on ImageNet-1k, ViT/Swin/ConvNeXt) and the full-test-set protocol, so the paper's absolute numbers are not comparable to ours.

## Differences from the round-53 version (`dynproto_r53`)

| Item | Round 53 | Now |
|---|---|---|
| ID prototypes | mean per predicted class | mean per true training label (§III-D) |
| After cold start, admission | Otsu on a 64-bin histogram of the batch scores | Eq. 5 / OWTTT grid search, weighted variance |
| Score used for admission | pre-update prototypes | pre-update prototypes (unchanged) |
| Score returned | pre-update prototypes; constant 1 while no prototype exists | post-update prototypes (Alg. 1 l.11-14); MSP while the cache is empty (l.16) |
| BIRCH | per changed class, `threshold=0.5`, caches of size 1 kept as is | same, plus explicit `branching_factor=50` |
| Output | `-S` (float64, saturates at τ = 0.01) | `-logit(S)` (same ranking, no saturation) |
| τ, K, m, β, T_cold | 0.01, 5, 30, 5, 5 | unchanged (K, m, β, T_cold match §IV-A; τ still our choice) |

## Smoke check

ImageNet-200 (ResNet-18), Textures, π = 0.05, seed 0 (4842 samples): AUROC 0.8763 (`dynproto_r53`) -> 0.9077 (`dynproto`, `alpha_rule="owttt"`), 0.9070 with `alpha_rule="eq5"`. Share of ID samples among cache entries used for scoring (DCR) 0.481 -> 0.420. Median `α` 0.34 (range 0.01-0.64).
