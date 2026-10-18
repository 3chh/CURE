# WARDEN: reimplementation notes

Paper: V. B. Balachandran, *Self-Poisoning in Adaptive Out-of-Distribution Detection: A Sharp-Threshold Theory and
Certified Label-Free Calibration*, arXiv 2607.21673 (JMLR submission, 2026). Local copy:
`papers/4_Test_time_OOD_detection_with_dynamic_memory/4.6_WARDEN_arXiv2026.pdf` (28 pages, read in full).

Official code: none. Section 9 says "the code and artifacts will be released publicly upon acceptance"; searches of
arXiv, GitHub (repositories and code) and the web on 2026-10-13 found no release. Everything below is reconstructed
from the text. The paper has no pseudo-code box for WARDEN; the only algorithm box is CDC (Section 5).

Code: `src/cure_ood/methods/testtime/warden.py`, registered as `warden` (faithful) and `warden_r53` (the earlier
approximation, kept only to reproduce round-53 tables). Tests: `tests/test_warden.py`.

## What WARDEN is (one paragraph)

WARDEN is a *decision* procedure wrapped around a frozen base detector, not a new ranking score. It ranks by the
frozen base score (kNN-to-bank on Ledoit-Wolf-whitened features), so "AUROC equals the frozen baseline by
construction" (Section 7, Detectors; Figure 5a; finding 3). It keeps an OOD dictionary whose admissions are certified
by e-BH on conformal p-values computed against a frozen reserve with the frozen scorer, never against the dictionary
(Section 4.3, Lemma 8, Theorem 9). It flags a point when either of two conformal channels fires at α/2: the base
channel or the dictionary-proximity channel (Section 7; Appendix D.1). CDC (Section 5) is the companion that
recalibrates the base threshold on the unlabelled stream under train-to-test drift. CDC is a threshold, not an alarm
or change detector, and it never changes the ranking.

## Component table

| Paper element | Code | Status |
|---|---|---|
| Base detector: kNN-to-bank (Sun et al. 2022), Section 7 "Settings" | `WARDEN.base_score` (l.119), `kth_dist` (l.75) against `self._G` | Implemented: Euclidean distance to the k-th neighbour in the whitened bank. k = `bench.k` (paper ablates k ∈ {5,10,20}, default not stated) |
| Feature whitening, Ledoit-Wolf shrinkage, fit on train-ID only, Section 7 | `ledoit_wolf_whitener` (l.67), `WARDEN._embed` (l.102), `fit` (l.109) | Implemented (`whiten=True`). Order L2 (Sun et al.) → whiten, no re-normalisation, is our choice |
| Ranking score = base score; "AUROC equals the frozen baseline by construction", Section 7, Fig. 5a | `Output(s, trace)` (l.166) | Implemented. `whiten=False` gives a monotone transform of our shared kNN score (same AUROC) |
| Frozen reserve R of held-out ID, size m (m = 1500 at the operating point), Section 4.3 after Cor. 10 | `self._R` = whitened `bench.C` (l.115), `self._res_score` (l.116) | Implemented; m = number of calibration points (880 on ImageNet-200) |
| One-sided conformal p-value against R, Prop. 7 | `rank_p(s, self._res_score)` (l.128) | Implemented, (1 + #{r ≥ s})/(m + 1) |
| Admission e-value e_i = a·p_i^(a-1), a ∈ (0,1), Lemma 8 | `ebh_admit` (l.22) | Implemented, a = 0.1 |
| e-BH at level δ over the batch: k* = max{k: e_(k) ≥ K/(δk)}, App. A.5 | `ebh_admit` | Implemented, δ = 0.1, K = actual batch length (last batch shorter) |
| Absolute admission threshold p ≤ κ(δ) = (aδ)^(1/(1-a)), Lemma 8 | `admission_p_threshold` (l.36) | Implemented (diagnostic; implied by e-BH, checked in tests) |
| Binomial bound κ̄ and W̄ = Kκ̄, Lemma 8 | `kappa_bar` (l.41) | Implemented (diagnostic). Reproduces the paper's κ ≈ 0.006, κ̄ ≈ 0.0096, W̄ ≈ 0.62 |
| Severed feedback: admission evidence reads only reserve + frozen scorer, never the dictionary, Section 4.3, Fig. 1b | `score` l.154: `ebh_admit(p_base[idx], ...)` | Implemented; `p_base` does not depend on the dictionary |
| OOD dictionary, capacity N, eviction "literal FIFO (our implementation) evicts the oldest block", Prop. 6 | `dictionary = np.r_[dictionary, new][-self.cap:]` (l.157) | Implemented as point-level FIFO (a_t oldest points out for a_t admitted). N not stated; `cap=512` (same as our OODD host) |
| Dictionary-proximity channel C2, conformal against the frozen reserve, Section 7, App. D.1 | `score` l.138-143 | Implemented. Statistic = minus the Euclidean distance to the nearest dictionary point (whitened space); reserve statistic recomputed with the same dictionary. Exact proximity statistic not stated |
| Decision scorer is F_{t-1}-measurable (Prop. 7), CDC "decide-then-update" | `score`: channel uses D_{t-1}; CDC uses batches < t; admission after decision | Implemented. The old code admitted first and scored the same batch |
| Decision: flag if p_base ≤ α/2 or p_dict ≤ α/2, α = 0.1, Section 7 | `self.last_["flags"]` (l.161) | Implemented; empty dictionary → p_dict = 1 |
| CDC step 1: Storey π̂_λ, π̂_up = [π̂_λ + (1/(1-λ))·sqrt(log(2/η)/(2n))] clipped to [0,1] | `storey_upper` (l.51) | Implemented, λ = 1/2 |
| CDC step 2: τ̂ = F̂_mix⁻¹(1 - α(1 - π̂_up) + ε_n), ε_n = sqrt(log(2/η)/(2n)) + 1/n; level ≥ 1 → τ̂ = +∞ (Thm. 11) | `cdc_threshold` (l.58) | Implemented; F̂⁻¹ = left-continuous empirical quantile |
| CDC step 3: refresh window; batch t uses batches < t | `score` l.145-152 | Implemented as a sliding window of the last `cdc_window` = 1024 stream points (window size n not stated) |
| CDC reserve = stale train-ID scores R_tr | same reserve `self._res_score` | Our C is held-out ID from the same split, so drift is small in our protocol |
| CDC p-values: conformal against the stale reserve | `p_base[lo:st]` | Implemented |
| How CDC composes with WARDEN's two channels | `last_["flags_cdc"]` (CDC alone at α, as in the paper's experiment rows) and `last_["flags_cdc_dict"]` (our composition: CDC threshold at α/2 on the base channel OR dictionary channel at α/2) | Paper says only that "the two components compose" (Section 4.3); composition is our choice |
| Prop. 14 block-dependent slack ε(n_eff) | not implemented | Paper itself evaluates i.i.d.-calibrated CDC ("no inflation") and calls the correction not needed in practice |
| Theorems 3, 9, 11-13, 15; Cor. 4, 10; Prop. 5, 6 | theory only | Nothing to implement; Lemma 8 constants are exposed for checking |
| Affine-refresh ablation, AdaODD/OODD ungated baselines, tail-mimicry attack (App. D.2) | not part of WARDEN | Out of scope for this file (OODD/AdaODD exist separately) |
| Batch handling: batches of K = 64, Section 4.3 operating point | `batch` argument | Default 128 (our protocol); `batch=64` gives the paper's K |

## Hyper-parameters

The paper uses one operating point for every setting; there are no per-dataset values.

| Parameter | Paper | Code default | Note |
|---|---|---|---|
| a (e-value exponent, called κ in the old code) | 0.1 | `a=0.1` | Lemma 8 operating point |
| δ (e-BH level) | 0.1 | `delta=0.1` | ablated over [0.05, 0.3] |
| α (decision FPR) | 0.1 | `alpha=0.1` | ablated over {0.05, 0.1} |
| K (batch) | 64 | `batch=128` | forced by our protocol |
| m (reserve size) | 1500 | |C| | given by the benchmark |
| η | 0.05 (Lemma 8) | `eta=0.05` | CDC η not stated separately; same value used |
| λ (Storey) | 1/2 | `lam=0.5` | |
| k (kNN) | {5, 10, 20} ablated, default not stated | `bench.k` | our benchmark K |
| N (dictionary cap) | not stated | `cap=512` | |
| CDC window n | not stated (Prop. 14 example n = 4000) | `cdc_window=1024` | |
| whitening | Ledoit-Wolf on train-ID | `whiten=True` on `bench.train_feats` | |

There is no recommended grid. The old `WARDEN_GRID` (three (κ, δ) pairs, best picked by test AUROC) is not from the
paper. It survives only as `R53_GRID` for the `warden_r53` arm so that the published round-53 tables still reproduce.
Picking the best of three configurations on test labels is an optimistic upper bound and must not be the headline
WARDEN number.

## Evaluation protocol of the paper vs ours

| Paper (Section 3, 7) | Ours |
|---|---|
| 96 cached-feature settings (vision, foundation, text, document), base AUROC in [0.62, 0.995] | OpenOOD v1.5 benchmarks (CIFAR-10/100, ImageNet-200, SupCon, DINOv2) |
| π ∈ {0.01, 0.05, 0.1, 0.5}, orders i.i.d. and bursty, 5 seeds + 5 held-out seeds | π ∈ {0.05, 0.2, 0.5}, i.i.d. shuffle, seeds 0-2 |
| Metrics: realized FPR at α, TPR, AUROC on the base score, dictionary impurity | Runner reports AUROC/AUPR/FPR95/TPR@5 of `Output.scores` and DCR (= dictionary impurity) from `Output.memory`; WARDEN's flags are in `method.last_` but the runner does not log them |
| Frozen backbone, cached features | Same (forced: we never touch the backbone) |
| Reserve = held-out eval-ID, stale reserve = train-ID for CDC | Reserve = calibration split C for both |

## Unspecified points and choices

1. Kernel of the dictionary-proximity statistic (1-NN, k-NN, cosine or Euclidean): 1-NN Euclidean distance in whitened space.
2. Dictionary capacity N: 512. Eviction granularity: point-level FIFO (equal to "a_t oldest points out").
3. Whether C2 uses D_{t-1} or D_t: D_{t-1}, required by Prop. 7 for super-uniformity.
4. CDC window size and whether it slides or is refreshed in blocks: sliding window of 1024 past points.
5. CDC warm-up (first batch, no past window): static reserve threshold p ≤ α.
6. How CDC and the two-channel decision compose: both exposed; composed variant is ours.
7. Whitening order: L2-normalise (Sun et al.), then whiten, then Euclidean kNN. Re-normalising after whitening was
   rejected: after centring, L2 discards the norm and inverted AUROC on a synthetic check (0.17).
8. k of kNN: benchmark K.

## Deviations forced by our protocol

- Input is cached penultimate features of a frozen backbone (the paper does the same).
- Batches of 128 instead of K = 64; e-BH uses K = 128 in the threshold K/(δk).
- Reserve size m = |C| (880 on ImageNet-200), so the smallest p is 1/(m+1) ≈ 0.0011 and admission needs at least
  ~29 points per batch of 128 at p = 1/(m+1) (e = 0.1·881^0.9 ≈ 45 ≥ 1280/k). Small OOD fractions are rarely admitted.
- The runner only scores rankings; FPR guarantees of WARDEN/CDC are available through `last_` but are not in tables.

## Smoke result (ImageNet-200, Textures, π = 0.05, seed 0)

| Arm | AUROC |
|---|---|
| kNN (our protocol) | 0.9472 |
| `warden_r53`, best of 3 configs on test labels (old) | 0.9472 (no admissions at π = 0.05; round-53 file agrees) |
| `warden`, paper default (whitened kNN ranking) | 0.9648 |
| `warden`, `whiten=False` | 0.9472 |

Decisions of `warden` on this stream: 0 admissions; two-channel flags FPR 0.046, TPR 0.843; CDC alone FPR 0.015,
TPR 0.645. At π = 0.5: 1341 admissions, dictionary impurity 0.004, flags FPR 0.092, TPR 0.866; CDC is very
conservative there (FPR 0.001, TPR 0.020), the power price of Theorem 15 at high contamination.

## Consequence for our tables

Because WARDEN ranks by the base score, its AUROC is exactly the base detector's (whitened kNN). The gain over our kNN
comes from Ledoit-Wolf whitening, not from the memory. The old arm
`warden_best` differed from kNN only because it ranked by min(p_base, p_dict), which the paper does not do, and then
picked the best of three settings on test labels.
