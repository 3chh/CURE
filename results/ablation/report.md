# ablation

135 streams. Metrics in %, averaged over seeds per group (benchmark, OOD set, π), then over groups. FPR95 lower is better; all others higher is better.

## Main table (45 groups)

| Method | AUROC | FPR95 | Safe groups | Mean rank |
|---|---|---|---|---|
| kNN | 86.19 | 43.2 | 100.0% | 7.93 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | 89.96 | 34.5 | 100.0% | 4.22 |
| No review (every expired sample enters CDM) | 85.40 | 53.4 | 57.8% | 7.62 |
| Oracle review (true labels, upper bound) | 92.76 | 26.0 | 91.1% | 2.42 |
| No fallback | 71.12 | 74.9 | 33.3% | 10.36 |
| Fisher combination | 89.95 | 34.4 | 100.0% | 4.16 |
| e-LOND certifier | 88.78 | 36.6 | 100.0% | 6.24 |
| Window 128 | 89.11 | 36.1 | 97.8% | 6.16 |
| Window 512 | 90.13 | 34.3 | 97.8% | 5.11 |
| Capacity 128 | 89.93 | 34.7 | 97.8% | 5.71 |
| Capacity 2048 | 89.98 | 34.4 | 100.0% | 6.07 |

Safe group: AUROC no more than 0.5 below kNN. Ties in rank are broken by row order.

## Same-base comparison: kNN

### all (45 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kNN | 86.19 | 94.21 | 57.30 | 43.15 | 39.31 | - | - | +0.00 | 0.0% | +0.00 | 7.93 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | 89.96 | 96.02 | 67.03 | 34.47 | 56.32 | 11.0% | 133 | +3.76 | 0.0% | -0.40 | 4.22 |
| No review (every expired sample enters CDM) | 85.40 | 93.50 | 62.77 | 53.38 | 49.89 | 75.0% | 495 | -0.80 | 42.2% | -22.33 | 7.62 |
| Oracle review (true labels, upper bound) | 92.76 | 97.08 | 73.78 | 25.99 | 64.87 | 0.0% | 381 | +6.57 | 8.9% | -1.55 | 2.42 |
| No fallback | 71.12 | 87.46 | 52.78 | 74.94 | 40.72 | 11.0% | 133 | -15.08 | 66.7% | -44.24 | 10.36 |
| Fisher combination | 89.95 | 96.05 | 66.34 | 34.44 | 56.68 | 11.0% | 133 | +3.76 | 0.0% | -0.32 | 4.16 |
| e-LOND certifier | 88.78 | 95.76 | 62.64 | 36.57 | 51.51 | 1.9% | 89 | +2.59 | 0.0% | -0.00 | 6.24 |
| Window 128 | 89.11 | 95.82 | 63.81 | 36.07 | 52.82 | 4.5% | 109 | +2.92 | 2.2% | -0.93 | 6.16 |
| Window 512 | 90.13 | 96.05 | 67.80 | 34.26 | 57.27 | 8.4% | 151 | +3.94 | 2.2% | -1.38 | 5.11 |
| Capacity 128 | 89.93 | 95.91 | 67.27 | 34.65 | 56.68 | 8.5% | 44 | +3.74 | 2.2% | -1.88 | 5.71 |
| Capacity 2048 | 89.98 | 96.04 | 67.05 | 34.37 | 56.51 | 10.9% | 342 | +3.79 | 0.0% | -0.40 | 6.07 |

Paired comparison (positive = CURE better; FPR95 sign flipped; 95% bootstrap CI over groups; one-sided Wilcoxon).

| Rival | Metric | Diff | 95% CI | Win / loss | p |
|---|---|---|---|---|---|
| No review (every expired sample enters CDM) | AUROC | +4.56 | [+2.42, +6.93] | 33 / 12 | 2.0e-04 |
| No review (every expired sample enters CDM) | FPR95 | +18.91 | [+10.67, +27.83] | 31 / 13 | 4.8e-04 |
| No review (every expired sample enters CDM) | AUPR-Out | +4.26 | [+2.25, +6.43] | 35 / 10 | 1.2e-04 |
| No review (every expired sample enters CDM) | TPR@FPR5 | +6.42 | [+3.85, +9.02] | 37 / 8 | 8.7e-06 |
| Oracle review (true labels, upper bound) | AUROC | -2.80 | [-3.84, -1.82] | 8 / 37 | 1.0e+00 |
| Oracle review (true labels, upper bound) | FPR95 | -8.48 | [-11.24, -5.87] | 5 / 38 | 1.0e+00 |
| Oracle review (true labels, upper bound) | AUPR-Out | -6.75 | [-9.43, -4.35] | 6 / 38 | 1.0e+00 |
| Oracle review (true labels, upper bound) | TPR@FPR5 | -8.55 | [-11.52, -5.80] | 7 / 38 | 1.0e+00 |
| No fallback | AUROC | +18.84 | [+14.34, +23.53] | 45 / 0 | 2.8e-14 |
| No fallback | FPR95 | +40.47 | [+32.44, +48.55] | 44 / 0 | 3.8e-09 |
| No fallback | AUPR-Out | +14.24 | [+9.80, +19.14] | 45 / 0 | 2.8e-14 |
| No fallback | TPR@FPR5 | +15.59 | [+10.13, +21.35] | 43 / 2 | 2.0e-13 |
| Fisher combination | AUROC | +0.00 | [-0.10, +0.10] | 11 / 19 | 5.9e-01 |
| Fisher combination | FPR95 | -0.03 | [-0.27, +0.19] | 17 / 13 | 3.5e-01 |
| Fisher combination | AUPR-Out | +0.68 | [-0.09, +1.71] | 14 / 16 | 2.7e-01 |
| Fisher combination | TPR@FPR5 | -0.37 | [-1.06, +0.20] | 12 / 17 | 8.1e-01 |
| e-LOND certifier | AUROC | +1.17 | [+0.35, +2.20] | 23 / 6 | 2.9e-04 |
| e-LOND certifier | FPR95 | +2.10 | [+0.46, +4.14] | 18 / 11 | 1.5e-02 |
| e-LOND certifier | AUPR-Out | +4.38 | [+1.09, +8.64] | 23 / 6 | 1.7e-04 |
| e-LOND certifier | TPR@FPR5 | +4.80 | [+1.42, +9.10] | 23 / 7 | 1.5e-04 |
| Window 128 | AUROC | +0.85 | [+0.26, +1.63] | 21 / 8 | 1.6e-03 |
| Window 128 | FPR95 | +1.60 | [+0.40, +3.14] | 17 / 12 | 8.9e-03 |
| Window 128 | AUPR-Out | +3.22 | [+0.76, +6.67] | 20 / 10 | 1.1e-03 |
| Window 128 | TPR@FPR5 | +3.50 | [+0.99, +6.96] | 19 / 11 | 1.6e-03 |
| Window 512 | AUROC | -0.18 | [-0.31, -0.05] | 9 / 22 | 9.9e-01 |
| Window 512 | FPR95 | -0.21 | [-0.70, +0.26] | 13 / 18 | 9.1e-01 |
| Window 512 | AUPR-Out | -0.78 | [-1.43, -0.27] | 9 / 22 | 1.0e+00 |
| Window 512 | TPR@FPR5 | -0.95 | [-1.59, -0.38] | 9 / 22 | 1.0e+00 |
| Capacity 128 | AUROC | +0.03 | [-0.15, +0.21] | 15 / 18 | 4.4e-01 |
| Capacity 128 | FPR95 | +0.18 | [-0.53, +0.93] | 17 / 16 | 3.8e-01 |
| Capacity 128 | AUPR-Out | -0.25 | [-0.65, +0.10] | 15 / 18 | 7.9e-01 |
| Capacity 128 | TPR@FPR5 | -0.36 | [-1.34, +0.53] | 16 / 17 | 6.7e-01 |
| Capacity 2048 | AUROC | -0.02 | [-0.05, -0.00] | 1 / 12 | 9.9e-01 |
| Capacity 2048 | FPR95 | -0.10 | [-0.21, -0.03] | 4 / 9 | 9.9e-01 |
| Capacity 2048 | AUPR-Out | -0.02 | [-0.06, +0.00] | 4 / 9 | 9.6e-01 |
| Capacity 2048 | TPR@FPR5 | -0.19 | [-0.42, -0.04] | 1 / 11 | 1.0e+00 |

### near (18 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kNN | 83.49 | 92.77 | 52.67 | 52.10 | 33.02 | - | - | +0.00 | 0.0% | +0.00 | 5.61 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | 83.94 | 93.00 | 53.59 | 52.09 | 35.07 | 20.1% | 8 | +0.44 | 0.0% | -0.40 | 4.00 |
| No review (every expired sample enters CDM) | 79.54 | 90.26 | 50.72 | 66.65 | 32.27 | 75.0% | 493 | -3.95 | 44.4% | -22.33 | 5.61 |
| Oracle review (true labels, upper bound) | 87.97 | 95.08 | 59.98 | 41.75 | 43.45 | 0.0% | 371 | +4.48 | 22.2% | -1.55 | 3.50 |
| No fallback | 56.63 | 79.50 | 32.76 | 98.87 | 14.19 | 20.1% | 8 | -26.87 | 94.4% | -41.93 | 10.89 |
| Fisher combination | 83.96 | 93.02 | 53.62 | 52.12 | 35.10 | 20.1% | 8 | +0.47 | 0.0% | -0.32 | 4.17 |
| e-LOND certifier | 83.64 | 92.86 | 52.92 | 52.02 | 33.50 | 0.0% | 0 | +0.14 | 0.0% | -0.00 | 5.83 |
| Window 128 | 83.61 | 92.75 | 52.98 | 52.52 | 33.93 | 2.5% | 1 | +0.11 | 5.6% | -0.93 | 6.78 |
| Window 512 | 84.07 | 93.01 | 53.95 | 52.26 | 35.63 | 13.1% | 22 | +0.58 | 5.6% | -1.38 | 5.50 |
| Capacity 128 | 83.84 | 92.88 | 53.63 | 52.62 | 35.12 | 21.0% | 9 | +0.34 | 5.6% | -1.88 | 6.56 |
| Capacity 2048 | 83.94 | 93.00 | 53.59 | 52.09 | 35.07 | 20.1% | 8 | +0.44 | 0.0% | -0.40 | 7.56 |

### far (27 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kNN | 87.99 | 95.16 | 60.38 | 37.19 | 43.50 | - | - | +0.00 | 0.0% | +0.00 | 9.48 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | 93.97 | 98.03 | 75.98 | 22.73 | 70.48 | 7.6% | 216 | +5.98 | 0.0% | +0.00 | 4.37 |
| No review (every expired sample enters CDM) | 89.30 | 95.66 | 70.80 | 44.53 | 61.65 | 75.0% | 496 | +1.31 | 40.7% | -18.46 | 8.96 |
| Oracle review (true labels, upper bound) | 95.95 | 98.41 | 82.97 | 15.48 | 79.15 | 0.0% | 387 | +7.96 | 0.0% | +1.09 | 1.70 |
| No fallback | 80.78 | 92.76 | 66.13 | 58.99 | 58.41 | 7.6% | 216 | -7.21 | 48.1% | -44.24 | 10.00 |
| Fisher combination | 93.95 | 98.07 | 74.83 | 22.66 | 71.07 | 7.6% | 216 | +5.96 | 0.0% | +0.00 | 4.15 |
| e-LOND certifier | 92.21 | 97.70 | 69.13 | 26.27 | 63.52 | 2.0% | 148 | +4.22 | 0.0% | -0.00 | 6.52 |
| Window 128 | 92.78 | 97.87 | 71.02 | 25.11 | 65.41 | 4.9% | 181 | +4.79 | 0.0% | +0.00 | 5.74 |
| Window 512 | 94.17 | 98.08 | 77.04 | 22.26 | 71.69 | 6.7% | 237 | +6.18 | 0.0% | +0.00 | 4.85 |
| Capacity 128 | 93.99 | 97.93 | 76.37 | 22.68 | 71.05 | 2.9% | 68 | +6.00 | 0.0% | +0.00 | 5.15 |
| Capacity 2048 | 94.00 | 98.07 | 76.02 | 22.55 | 70.80 | 7.6% | 564 | +6.01 | 0.0% | +0.00 | 5.07 |

### kNN: by benchmark and π (AUROC / FPR95)

| Part | Groups | kNN | **CURE** (e-BH, Cauchy, window 256, capacity 512) | No review (every expired sample enters CDM) | Oracle review (true labels, upper bound) | No fallback | Fisher combination | e-LOND certifier | Window 128 | Window 512 | Capacity 128 | Capacity 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar100 | 18 | 81.04 / 54.6 | 88.59 / 36.5 | 88.06 / 36.0 | 92.33 / 24.6 | 73.55 / 67.4 | 88.35 / 36.9 | 86.07 / 41.4 | 86.90 / 39.9 | 88.85 / 36.2 | 88.42 / 37.2 | 88.61 / 36.4 |
| cifar100 near | 6 | 80.41 / 58.3 | 80.61 / 58.1 | 82.48 / 49.3 | 86.61 / 39.7 | 52.88 / 99.0 | 80.60 / 58.2 | 80.41 / 58.3 | 80.41 / 58.3 | 80.91 / 58.0 | 80.54 / 58.4 | 80.61 / 58.1 |
| cifar100 far | 12 | 81.35 / 52.8 | 92.58 / 25.7 | 90.84 / 29.3 | 95.19 / 17.1 | 83.88 / 51.7 | 92.22 / 26.3 | 88.90 / 33.0 | 90.14 / 30.6 | 92.82 / 25.3 | 92.36 / 26.6 | 92.61 / 25.5 |
| cifar10 | 18 | 92.40 / 27.3 | 93.75 / 24.5 | 81.91 / 74.4 | 94.53 / 20.7 | 71.67 / 72.1 | 93.96 / 24.0 | 93.61 / 24.5 | 93.58 / 24.8 | 93.83 / 24.4 | 93.83 / 24.2 | 93.78 / 24.4 |
| cifar10 near | 6 | 91.12 / 33.5 | 91.06 / 34.6 | 72.77 / 95.0 | 90.33 / 35.6 | 57.03 / 99.9 | 91.12 / 34.6 | 91.12 / 33.5 | 90.96 / 34.8 | 90.90 / 35.4 | 90.78 / 35.9 | 91.06 / 34.6 |
| cifar10 far | 12 | 93.05 / 24.2 | 95.09 / 19.5 | 86.49 / 64.1 | 96.63 / 13.3 | 78.99 / 58.2 | 95.38 / 18.7 | 94.85 / 20.0 | 94.89 / 19.7 | 95.29 / 19.0 | 95.35 / 18.4 | 95.14 / 19.2 |
| imagenet200 | 9 | 84.08 / 51.9 | 85.10 / 50.3 | 87.04 / 46.2 | 90.08 / 39.2 | 65.15 / 95.6 | 85.15 / 50.3 | 84.55 / 51.0 | 84.58 / 51.1 | 85.31 / 50.0 | 85.15 / 50.4 | 85.10 / 50.3 |
| imagenet200 near | 6 | 78.95 / 64.5 | 80.14 / 63.5 | 83.38 / 55.6 | 86.98 / 50.0 | 59.97 / 97.8 | 80.16 / 63.6 | 79.37 / 64.3 | 79.45 / 64.5 | 80.40 / 63.4 | 80.19 / 63.6 | 80.14 / 63.5 |
| imagenet200 far | 3 | 94.35 / 26.8 | 95.02 / 23.9 | 94.37 / 27.2 | 96.29 / 17.6 | 75.50 / 91.2 | 95.13 / 23.8 | 94.90 / 24.5 | 94.85 / 24.5 | 95.12 / 23.3 | 95.06 / 23.9 | 95.03 / 23.9 |
| π = 0.05 | 15 | 86.10 / 43.4 | 88.03 / 39.8 | 81.35 / 64.1 | 92.13 / 27.8 | 55.73 / 99.9 | 87.95 / 39.9 | 86.10 / 43.4 | 86.54 / 42.8 | 88.23 / 39.1 | 88.17 / 39.0 | 88.03 / 39.8 |
| π = 0.2 | 15 | 86.23 / 43.1 | 90.41 / 33.0 | 85.60 / 52.6 | 92.93 / 25.5 | 74.06 / 70.5 | 90.42 / 32.9 | 89.20 / 35.1 | 89.67 / 34.2 | 90.58 / 32.9 | 90.46 / 33.0 | 90.41 / 32.9 |
| π = 0.5 | 15 | 86.24 / 43.0 | 91.43 / 30.7 | 89.24 / 43.4 | 93.21 / 24.6 | 83.57 / 54.4 | 91.48 / 30.6 | 91.05 / 31.2 | 91.12 / 31.3 | 91.58 / 30.8 | 91.16 / 31.9 | 91.49 / 30.4 |

## ID/OOD decisions (flag OOD when p ≤ α)

Realised FPR on ID and TPR on OOD in %. p is conformal against the calibration set, so FPR should stay near α or below when the ID part of the stream is exchangeable with it.

| Method | Groups | α | FPR (mean) | FPR (worst group) | TPR |
|---|---|---|---|---|---|
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | 45 | 0.05 | 5.02 | 5.96 | 56.46 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | 45 | 0.1 | 10.15 | 11.80 | 69.75 |
| Window 128 | 45 | 0.05 | 4.97 | 5.64 | 52.77 |
| Window 128 | 45 | 0.1 | 10.19 | 11.66 | 67.15 |
| Capacity 128 | 45 | 0.05 | 4.98 | 5.68 | 56.61 |
| Capacity 128 | 45 | 0.1 | 10.20 | 11.69 | 70.06 |
| e-LOND certifier | 45 | 0.05 | 4.96 | 5.55 | 51.44 |
| e-LOND certifier | 45 | 0.1 | 10.22 | 11.71 | 66.20 |
| Fisher combination | 45 | 0.05 | 5.07 | 5.78 | 56.93 |
| Fisher combination | 45 | 0.1 | 10.17 | 11.83 | 70.26 |
| No fallback | 45 | 0.05 | 2.66 | 5.80 | 37.89 |
| No fallback | 45 | 0.1 | 5.26 | 10.81 | 42.61 |
| Capacity 2048 | 45 | 0.05 | 5.01 | 5.92 | 56.64 |
| Capacity 2048 | 45 | 0.1 | 10.12 | 11.80 | 69.82 |
| Window 512 | 45 | 0.05 | 5.03 | 5.93 | 57.42 |
| Window 512 | 45 | 0.1 | 10.18 | 11.66 | 70.48 |
| No review (every expired sample enters CDM) | 45 | 0.05 | 4.99 | 5.68 | 49.81 |
| No review (every expired sample enters CDM) | 45 | 0.1 | 10.11 | 10.93 | 62.74 |
| Oracle review (true labels, upper bound) | 45 | 0.05 | 4.98 | 5.75 | 64.64 |
| Oracle review (true labels, upper bound) | 45 | 0.1 | 10.03 | 11.60 | 77.17 |

## CDM activity

CDM: mean CDM size per batch. Coverage: share of the stream scored with a non-empty memory. Certified / evicted: samples per stream that left the review queue into CDM / out of the system.

| Method | Part | Groups | CDM | Coverage | Certified | Evicted |
|---|---|---|---|---|---|---|
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | all | 45 | 129 | 52.3% | 1102 | 9753 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | near | 18 | 8 | 23.6% | 14 | 10042 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | far | 27 | 210 | 71.4% | 1828 | 9561 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | π = 0.05 | 15 | 1 | 14.7% | 2 | 8268 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | π = 0.2 | 15 | 110 | 59.0% | 444 | 9336 |
| **CURE** (e-BH, Cauchy, window 256, capacity 512) | π = 0.5 | 15 | 277 | 83.3% | 2861 | 11655 |
| Window 128 | all | 45 | 107 | 41.0% | 906 | 10077 |
| Window 128 | near | 18 | 1 | 13.8% | 2 | 10182 |
| Window 128 | far | 27 | 178 | 59.1% | 1509 | 10007 |
| Window 128 | π = 0.05 | 15 | 0 | 3.0% | 0 | 8398 |
| Window 128 | π = 0.2 | 15 | 86 | 44.6% | 352 | 9557 |
| Window 128 | π = 0.5 | 15 | 235 | 75.4% | 2367 | 12277 |
| Capacity 128 | all | 45 | 43 | 56.1% | 690 | 2826 |
| Capacity 128 | near | 18 | 8 | 31.8% | 16 | 3243 |
| Capacity 128 | far | 27 | 65 | 72.4% | 1140 | 2549 |
| Capacity 128 | π = 0.05 | 15 | 3 | 17.2% | 5 | 2673 |
| Capacity 128 | π = 0.2 | 15 | 43 | 61.2% | 377 | 2792 |
| Capacity 128 | π = 0.5 | 15 | 82 | 90.0% | 1689 | 3014 |
| e-LOND certifier | all | 45 | 86 | 32.3% | 772 | 10084 |
| e-LOND certifier | near | 18 | 0 | 3.5% | 0 | 10056 |
| e-LOND certifier | far | 27 | 144 | 51.5% | 1286 | 10102 |
| e-LOND certifier | π = 0.05 | 15 | 0 | 0.0% | 0 | 8270 |
| e-LOND certifier | π = 0.2 | 15 | 76 | 34.2% | 321 | 9460 |
| e-LOND certifier | π = 0.5 | 15 | 183 | 62.7% | 1994 | 12522 |
| Fisher combination | all | 45 | 129 | 52.3% | 1102 | 9753 |
| Fisher combination | near | 18 | 8 | 23.6% | 14 | 10042 |
| Fisher combination | far | 27 | 210 | 71.4% | 1828 | 9561 |
| Fisher combination | π = 0.05 | 15 | 1 | 14.7% | 2 | 8268 |
| Fisher combination | π = 0.2 | 15 | 110 | 59.0% | 444 | 9336 |
| Fisher combination | π = 0.5 | 15 | 277 | 83.3% | 2861 | 11655 |
| No fallback | all | 45 | 129 | 52.3% | 1102 | 9753 |
| No fallback | near | 18 | 8 | 23.6% | 14 | 10042 |
| No fallback | far | 27 | 210 | 71.4% | 1828 | 9561 |
| No fallback | π = 0.05 | 15 | 1 | 14.7% | 2 | 8268 |
| No fallback | π = 0.2 | 15 | 110 | 59.0% | 444 | 9336 |
| No fallback | π = 0.5 | 15 | 277 | 83.3% | 2861 | 11655 |
| Capacity 2048 | all | 45 | 333 | 52.3% | 1102 | 9753 |
| Capacity 2048 | near | 18 | 8 | 23.6% | 14 | 10042 |
| Capacity 2048 | far | 27 | 550 | 71.4% | 1828 | 9561 |
| Capacity 2048 | π = 0.05 | 15 | 1 | 14.7% | 2 | 8268 |
| Capacity 2048 | π = 0.2 | 15 | 211 | 59.0% | 444 | 9336 |
| Capacity 2048 | π = 0.5 | 15 | 788 | 83.3% | 2861 | 11655 |
| Window 512 | all | 45 | 144 | 56.9% | 1221 | 7179 |
| Window 512 | near | 18 | 21 | 33.2% | 42 | 7723 |
| Window 512 | far | 27 | 226 | 72.6% | 2007 | 6816 |
| Window 512 | π = 0.05 | 15 | 3 | 17.4% | 8 | 6347 |
| Window 512 | π = 0.2 | 15 | 132 | 61.5% | 520 | 7028 |
| Window 512 | π = 0.5 | 15 | 297 | 91.7% | 3135 | 8162 |
| No review (every expired sample enters CDM) | all | 45 | 482 | 97.4% | 10855 | 0 |
| No review (every expired sample enters CDM) | near | 18 | 479 | 97.1% | 10056 | 0 |
| No review (every expired sample enters CDM) | far | 27 | 484 | 97.6% | 11388 | 0 |
| No review (every expired sample enters CDM) | π = 0.05 | 15 | 475 | 96.8% | 8270 | 0 |
| No review (every expired sample enters CDM) | π = 0.2 | 15 | 481 | 97.3% | 9780 | 0 |
| No review (every expired sample enters CDM) | π = 0.5 | 15 | 491 | 98.1% | 14516 | 0 |
| Oracle review (true labels, upper bound) | all | 45 | 369 | 96.4% | 3209 | 7647 |
| Oracle review (true labels, upper bound) | near | 18 | 358 | 96.1% | 2978 | 7078 |
| Oracle review (true labels, upper bound) | far | 27 | 375 | 96.6% | 3363 | 8025 |
| Oracle review (true labels, upper bound) | π = 0.05 | 15 | 199 | 95.4% | 414 | 7856 |
| Oracle review (true labels, upper bound) | π = 0.2 | 15 | 426 | 96.2% | 1955 | 7826 |
| Oracle review (true labels, upper bound) | π = 0.5 | 15 | 481 | 97.6% | 7258 | 7258 |
