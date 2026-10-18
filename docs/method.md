# CURE in one page

Code: `src/cure_ood/methods/cure.py`, `methods/memory.py`, `stats.py`. Defaults below are the official configuration.

## Inputs (prepared once, ID data only)

| Symbol | Meaning | Default |
|---|---|---|
| G | gallery: L2-normalised ID training features | ID train split |
| C | calibration set: held-out ID samples, never in the stream | ID val split |
| s0(x) | static score: cosine to the K-th neighbour in G (larger = more ID) | K per benchmark (APS) |
| p_s(x) | conformal p-value of s0 against C: (1 + #{c: s0(c) ≤ s0(x)}) / (\|C\| + 1) | |
| τ | screening threshold: ρ-quantile of s0 on C | ρ = 0.2 |
| S | reference set: {c ∈ C : s0(c) ≤ τ}, the ID tail | |

## Per batch (B = 128)

1. **Retroactive review.** Every sample enters a review queue when it arrives. Samples that have waited `window` = 256 samples are reviewed:
   - radius r_e = cosine from e to its k-th neighbour in S (k = 10);
   - N_e = number of later screened samples (s0 ≤ τ) inside that radius, among the `window` samples after e;
   - under the null "e and the counted samples are ID tail, exchangeable with S", N_e ~ BetaBinomial(n_pass, k, |S| − k + 1), giving p_e.
2. **Certification.** e-values e = κ p_e^(κ−1) (κ = 0.5), e-BH at level α = 0.2 over the samples expiring in this batch. Certified samples enter the curated dynamic memory (CDM, FIFO, 512); the rest are evicted. Evicted samples were never used for scoring.
3. **Memory channel.** If CDM is non-empty: a = cosine to the k_mem-th neighbour in CDM (k_mem = 5), statistic t = a − s0, p_mem = conformal p-value of t against the same statistic computed on C with the same CDM.
4. **Combination.** Cauchy combination of p_s and p_mem (weights 1:1), recalibrated on C in the same memory state (leave-one-out p-values on C). Score = −log p + 10⁻⁴·T − 10⁻⁷·s0 (T = combined statistic, tie-breakers only).
5. **Fallback.** If CDM is empty, score = −log p_s (the static detector).

Labels are never used. Changing the base detector (`base: msp | energy`) only replaces p_s and the fallback; screening, review and the memory channel keep using the kNN score.

## Variants as configurations

| Config | What changes |
|---|---|
| `cure` | official |
| `cure` with `base: msp` / `energy` | static channel and fallback use MSP / Energy |
| `cure_lf` (older version) | `certifier: elond`, `window: 512`, `wake: [256, 0.05]`, `memory: queue+cdm`, `rule: fisher`, `w_mem: 0.1`, `fallback: false` |
| ablations | e.g. `certifier`, `alpha`, `window`, `capacity`, `rule`, `w_mem` |
