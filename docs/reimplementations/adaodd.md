# AdaODD reimplementation

Paper: Y. Zhang, X. Wang, T. Zhou, K. Yuan, Z. Zhang, L. Wang, R. Jin, *Model-free Test Time Adaptation for
Out-Of-Distribution Detection*, arXiv:2311.16420 (v1, 28 Nov 2023; the only version on arXiv). 13 pages, no appendix
(the "Appendix VI-A" cited in Sec. III-A is Section VI-A).

Official code: none. The arXiv page has no code link, the paper has no code statement, and a web search
(GitHub, Papers with Code) finds no release. Everything below is reimplemented from the text.

Code: `src/cure_ood/methods/testtime/adaodd.py` (class `AdaODD`, helper `knn_score`, table `PRESETS`).
Tests: `tests/test_adaodd.py`.

## Paper element -> code

| Paper element | Code | Status |
|---|---|---|
| Memory bank initialisation `M = {(z_i = h(x_i)/‖h(x_i)‖, s_i = 1)}` over the ID training set (Alg. 1 l.2) | `fit`: `self.mem0 = bench.G` (L2 gallery); `score`: `wts = ones` | Implemented. "Training set" = our gallery split (see deviations). |
| Feature normalisation of the test sample (Alg. 1 l.4) | `stream.x` (L2 features) | Implemented. |
| Normalisation ablation, Fig. 6 / Fig. 10 (no L2 normalisation) | `normalize=False` uses `train_feats`, `cal_feats`, `stream.feats` | Implemented as variant. |
| k nearest neighbours by Euclidean distance (Alg. 1 l.5; Sec. VI-A "Implementation details": faiss, Euclidean) | `knn_score`: `torch.cdist(...).topk(k, largest=False)` | Implemented (exact search, torch instead of faiss). Neighbours are ranked by plain distance, the scale is applied afterwards. |
| Score `S(x) = (1/k) Σ -‖z - z_(i)‖₂ · s_(i)` (Alg. 1 l.6, Eq. 4) | `knn_score(..., agg="avg")` | Implemented. Output sign flipped (`-S`) because the framework wants OOD-positive scores. |
| Score variants k-th `-‖z - z_(k)‖ s_(k)` and k-median `-‖z - z_(⌊k/2⌋)‖ s_(⌊k/2⌋)` (Sec. VI-B "How to combine the score", Fig. 4, Fig. 9) | `agg="kth"`, `agg="median"` | Implemented as variants; `avg` is the paper default. |
| Decision `1{S(x) ≥ λ}` (Alg. 1 l.7) | `reevaluate` (accuracy uses `S ≥ λ`); the runner itself uses threshold-free metrics | Implemented where it matters (see protocol). |
| Threshold λ: "a high fraction of the ID validation data (95%) can be correctly classified" (Sec. III-B) | `fit`: `λ = quantile(S(C), 1 - tpr)`, `tpr=0.95`, memory = initial `M` | Implemented. |
| OOD selection: `S(x) < λγ` → `M ∪ {(z, κ)}` (Alg. 1 l.9-11, Eq. 6) | `score`: `add_ood = sc < lam*gamma`, weight `kappa` | Implemented, strict inequality as in the paper. |
| ID selection: `S(x) > λ/γ` → `M ∪ {(z, 1)}` (Alg. 1 l.12-14) | `score`: `add_id = sc > lam/gamma`, weight 1 | Implemented, strict inequality. |
| Constraints `γ ≥ 1`, `κ ≥ 1` (Sec. III-B) | `_resolve` raises `ValueError` | Implemented. |
| Order: score first, then augment memory (Alg. 1 l.3-14) | `score` loop | Implemented; per sample when `batch=1`, per batch otherwise. |
| "No requirement to update the indexing data structure for every individual test instance" (Sec. VI-C "Computational complexity") | `batch` | Used as the paper's own licence for batched updates. |
| Hyper-parameters CIFAR-10: k = 50, γ = 1.5, κ = 5.0 (Sec. VI-A "Hyper-parameters details") | `PRESETS["cifar10"]` | Implemented; also used for CIFAR-10 SupCon / DINOv2 (Table II uses one setting for both training losses). |
| CIFAR-100: k = 200 (Sec. VI-A) | `PRESETS["cifar100"]` | k implemented; γ, κ not given (see ambiguities). |
| ImageNet: k = 1000, γ = 1.0, κ = 10.0 (Sec. VI-A; Table VII γ = 1.0 best; Sec. VI-C "we choose κ = 10") | `PRESETS["imagenet"]` with `k=None` → 1‰ rule | γ, κ implemented; k replaced by the paper's rule (see deviations). |
| Rule "set k as 1‰ of the size of the training set" (Sec. VI-C "Effect of k") | `_resolve`: `k = max(1, round(N/1000))` when the preset gives none | Implemented. |
| Grids: k ∈ {1,10,20,50,100,200,500,1000,3000,5000}, γ ∈ {1.0,...,2.0}, κ ∈ {1,...,10,100}, selected with the OE validation method [11] | constructor args | Exposed as arguments; no selection is run (no OE validation OOD set in our protocol). |
| ID reevaluation accuracy (Sec. VI-B "Effect of TTA for ID data", Sec. VI-D, Fig. 5) | `AdaODD.reevaluate(feats)` | Implemented as a helper; not part of the runner metrics. |
| Algorithm 2 (synthetic, m labelled OOD samples inserted with κ = 100 before the test) | — | Not implemented: needs labelled OOD data, which the protocol forbids in `fit`. It is a toy experiment, not the method. |
| Log-ratio test Eq. 7 with ground-truth labels / 10% pre-selected OOD (Sec. V, Table I) | — | Not implemented: the paper itself rejects it; it needs true labels of memory entries. |
| TTA Energy / TTA Mahalanobis / TTA KNN baselines (Sec. VI-B, Table IV) | — | Out of scope (baselines of the paper, not AdaODD). |
| Evaluation: FPR95 at 95% ID TPR, AUROC, inference time (Sec. VI) | runner (`cure_ood.eval.metrics`) | AUROC/FPR95 computed by our runner; time not reported. |
| Evaluation settings: standard (one OOD set after ID), Sequential, OODs mixture, ID-OOD mixture (Sec. VI-D, Table VIII) | `Benchmark.stream` | Our stream is the paper's "ID-OOD mixture" setting with λ from held-out ID data, which is what the paper prescribes for it. |
| Covariate shift study (CIFAR-10-C noise, Table IX) | — | Not applicable to our benchmarks. |

## Points the paper leaves unspecified and our choice

1. **γ and κ defaults contradict each other.** Sec. VI-A gives γ = 1.5, κ = 5.0 for CIFAR-10; Sec. VI-B says
   "We choose κ = 1.5 and γ = 5.0 by default". The second pair lies outside both search grids
   (γ ∈ {1.0,...,2.0}, κ ∈ {1,...,10,100}) and outside the axes of Fig. 5, so we read it as a swap and use
   γ = 1.5, κ = 5.0. Round 53 used the literal sentence (k = 50, γ = 5.0, κ = 1.5); it remains reachable with
   `adaodd(k=50, gamma=5.0, kappa=1.5)` and reproduces round 53 bit-exactly (`tests/test_golden_testtime.py`).
2. **CIFAR-100 γ, κ** are not given. We use the CIFAR-10 values (same small-scale group in the paper).
3. **k for ImageNet-200.** The paper's k = 1000 is for a 1.28M-image memory. Our ImageNet-200 gallery has
   3,520 samples, so k = 1000 would average over 28% of the memory. We apply the paper's own rule
   k = 1‰ · |train| → k = 4. Pass `k=...` to override.
4. **Distance: squared or not.** The formula uses `‖·‖₂`; faiss `IndexFlatL2` returns squared distances. We follow
   the formula (unsquared).
5. **λ quantile.** "95% of ID validation correctly classified" is read as λ = 5% empirical quantile of S on the ID
   validation set (linear interpolation, `np.quantile`), computed once with the initial memory and never updated.
6. **Validation set.** The paper does not say which split is "ID validation" for ImageNet; we use the benchmark's
   calibration split `C` (disjoint from the gallery and from the stream).
7. **Index of the k-median.** "z_(⌊k/2⌋)" with 1-based ranks → 0-based index `max(k//2, 1) - 1`.
8. **Weights at test time for the λ computation.** All initial weights are 1, so λ does not depend on κ.
9. **A sample is never added twice.** S ≤ 0, so λ ≤ 0 and λγ ≤ λ ≤ λ/γ: the two conditions are disjoint. With
   γ = 1 (ImageNet preset) every sample except exact ties `S = λ` enters the memory, as OOD or as ID.
10. **Memory size.** The paper has no cap and no eviction; we keep it unbounded.

## Deviations forced by our protocol

- **Feature-level input, frozen backbone.** We start from pre-extracted OpenOOD v1.5 features; AdaODD never updates
  the backbone anyway, so this changes nothing in the method.
- **Backbones and datasets.** Paper: ResNet-18 (CIFAR-10, CE and SupCon), ResNet-50 / ViT-B/16 (ImageNet-1k).
  Ours: OpenOOD v1.5 ResNet-18 (CIFAR-10, CIFAR-100, ImageNet-200), SupCon, DINOv2.
- **Memory initialisation on ImageNet-200** uses our 3,520-sample gallery (half of the ID test set split by class)
  instead of the full ImageNet training set, which OpenOOD v1.5 does not ship for ImageNet-200.
- **Batches of 128.** Default `batch=128`: the whole batch is scored with the memory of previous batches, then the
  selected samples are appended. `batch=1` is the exact Alg. 1 order. On ImageNet-200 / Textures the two differ by
  less than 0.001 AUROC (0.9359 vs 0.9352 at π = 0.05; 0.9536 vs 0.9530 at π = 0.5).
- **Threshold-free metrics.** The runner reports AUROC/FPR95 over the scores; λ only drives memory updates.
- **No hyper-parameter search.** The paper selects k, γ, κ with an OE-style validation OOD set; we use the reported
  values and do not tune on our streams.

## Smoke check (ImageNet-200, Textures, seed 0)

| π | old (round 53 settings) | new default (k=4, γ=1.0, κ=10, batch 128) | new, batch=1 |
|---|---|---|---|
| 0.05 | 0.9292 | 0.9359 | 0.9352 |
| 0.5 | 0.9238 (= golden) | 0.9536 | 0.9530 |
