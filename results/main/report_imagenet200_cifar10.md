# main_imagenet200_cifar10

81 streams. Metrics in %, averaged over seeds per group (benchmark, OOD set, π), then over groups. FPR95 lower is better; all others higher is better.

## Main table (27 groups)

| Method | AUROC | FPR95 | Safe groups | Mean rank |
|---|---|---|---|---|
| kNN | 89.63 | 35.5 | 100.0% | 3.67 |
| OODD (official) | 85.00 | 44.8 | 22.2% | 5.81 |
| OODD + T-out (official) | 86.37 | 41.9 | 29.6% | 4.41 |
| AdaODD* (paper) | 90.23 | 36.1 | 96.3% | 2.74 |
| WARDEN* (paper) | 88.20 | 38.6 | 44.4% | 3.85 |
| Age-expiry queue | 63.60 | 75.6 | 25.9% | 5.48 |
| **CURE** | 90.87 | 33.1 | 100.0% | 2.04 |

Safe group: AUROC no more than 0.5 below kNN. Ties in rank are broken by row order.

## Same-base comparison: kNN

### all (27 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kNN | 89.63 | 95.47 | 64.72 | 35.53 | 50.79 | - | - | +0.00 | 0.0% | +0.00 | 3.81 |
| **CURE** | 90.87 | 96.11 | 68.10 | 33.12 | 58.79 | 5.4% | 113 | +1.24 | 0.0% | -0.40 | 2.04 |
| OODD (official) | 85.00 | 94.51 | 57.74 | 44.75 | 40.10 | 46.5% | 512 | -4.63 | 77.8% | -17.09 | 6.15 |
| OODD + T-out (official) | 86.37 | 94.83 | 59.72 | 41.87 | 43.01 | 44.9% | 463 | -3.26 | 70.4% | -12.56 | 4.41 |
| OODD (official, batch 128) | 84.98 | 94.49 | 57.62 | 45.16 | 40.44 | 46.5% | 501 | -4.65 | 77.8% | -16.55 | 6.48 |
| AdaODD* (paper) | 90.23 | 95.58 | 68.34 | 36.11 | 57.44 | - | - | +0.60 | 3.7% | -0.85 | 2.85 |
| WARDEN* (paper) | 88.20 | 94.72 | 65.16 | 38.62 | 52.42 | 0.5% | 15 | -1.43 | 55.6% | -8.63 | 4.11 |
| Age-expiry queue | 63.60 | 84.80 | 49.22 | 75.61 | 36.22 | 69.4% | 501 | -26.03 | 74.1% | -73.33 | 6.15 |

Paired comparison (positive = CURE better; FPR95 sign flipped; 95% bootstrap CI over groups; one-sided Wilcoxon).

| Rival | Metric | Diff | 95% CI | Win / loss | p |
|---|---|---|---|---|---|
| OODD (official) | AUROC | +5.86 | [+4.21, +7.69] | 27 / 0 | 7.5e-09 |
| OODD (official) | FPR95 | +11.63 | [+9.16, +14.08] | 26 / 1 | 1.5e-08 |
| OODD (official) | AUPR-Out | +10.36 | [+5.81, +15.69] | 25 / 2 | 9.4e-06 |
| OODD (official) | TPR@FPR5 | +18.68 | [+12.27, +26.09] | 26 / 1 | 2.5e-07 |
| OODD + T-out (official) | AUROC | +4.50 | [+3.10, +6.02] | 26 / 1 | 3.7e-08 |
| OODD + T-out (official) | FPR95 | +8.74 | [+6.69, +10.84] | 26 / 1 | 2.2e-08 |
| OODD + T-out (official) | AUPR-Out | +8.38 | [+3.72, +13.63] | 25 / 2 | 8.1e-05 |
| OODD + T-out (official) | TPR@FPR5 | +15.77 | [+9.21, +23.27] | 25 / 2 | 3.3e-06 |
| OODD (official, batch 128) | AUROC | +5.89 | [+4.37, +7.56] | 27 / 0 | 7.5e-09 |
| OODD (official, batch 128) | FPR95 | +12.04 | [+9.47, +14.59] | 26 / 1 | 1.5e-08 |
| OODD (official, batch 128) | AUPR-Out | +10.48 | [+6.07, +15.69] | 25 / 2 | 8.0e-06 |
| OODD (official, batch 128) | TPR@FPR5 | +18.35 | [+12.08, +25.60] | 26 / 1 | 2.5e-07 |
| AdaODD* (paper) | AUROC | +0.63 | [+0.12, +1.16] | 17 / 10 | 2.5e-02 |
| AdaODD* (paper) | FPR95 | +2.99 | [+1.02, +5.06] | 18 / 9 | 1.0e-02 |
| AdaODD* (paper) | AUPR-Out | -0.24 | [-2.19, +1.58] | 15 / 12 | 3.9e-01 |
| AdaODD* (paper) | TPR@FPR5 | +1.35 | [-0.66, +3.46] | 15 / 12 | 1.7e-01 |
| WARDEN* (paper) | AUROC | +2.67 | [+1.51, +3.97] | 21 / 6 | 2.7e-05 |
| WARDEN* (paper) | FPR95 | +5.50 | [+3.25, +7.65] | 22 / 5 | 5.5e-05 |
| WARDEN* (paper) | AUPR-Out | +2.94 | [+0.08, +5.50] | 20 / 7 | 8.7e-03 |
| WARDEN* (paper) | TPR@FPR5 | +6.37 | [+2.93, +9.67] | 21 / 6 | 9.3e-04 |
| Age-expiry queue | AUROC | +27.27 | [+17.47, +37.60] | 23 / 4 | 9.4e-06 |
| Age-expiry queue | FPR95 | +42.48 | [+29.36, +54.93] | 21 / 6 | 9.4e-06 |
| Age-expiry queue | AUPR-Out | +18.88 | [+12.18, +25.80] | 24 / 3 | 4.0e-06 |
| Age-expiry queue | TPR@FPR5 | +22.56 | [+14.70, +30.39] | 22 / 5 | 1.1e-05 |

### near (12 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kNN | 85.03 | 93.29 | 56.04 | 49.02 | 38.36 | - | - | +0.00 | 0.0% | +0.00 | 2.50 |
| **CURE** | 85.60 | 93.53 | 57.25 | 49.09 | 41.22 | 6.1% | 12 | +0.56 | 0.0% | -0.40 | 2.33 |
| OODD (official) | 77.69 | 91.27 | 45.82 | 59.59 | 22.57 | 56.3% | 512 | -7.34 | 100.0% | -15.45 | 6.83 |
| OODD + T-out (official) | 79.01 | 91.61 | 47.03 | 57.86 | 23.92 | 55.3% | 469 | -6.03 | 100.0% | -12.56 | 5.17 |
| OODD (official, batch 128) | 77.84 | 91.31 | 45.83 | 59.42 | 22.74 | 56.2% | 500 | -7.19 | 100.0% | -14.65 | 6.75 |
| AdaODD* (paper) | 85.57 | 93.26 | 58.45 | 48.55 | 42.37 | - | - | +0.53 | 0.0% | -0.10 | 2.17 |
| WARDEN* (paper) | 80.86 | 91.39 | 50.96 | 56.30 | 32.27 | - | - | -4.17 | 100.0% | -8.63 | 5.25 |
| Age-expiry queue | 58.85 | 81.97 | 40.13 | 76.08 | 22.71 | 69.8% | 500 | -26.18 | 66.7% | -73.33 | 5.00 |

### far (15 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kNN | 93.31 | 97.22 | 71.67 | 24.73 | 60.74 | - | - | +0.00 | 0.0% | +0.00 | 4.87 |
| **CURE** | 95.08 | 98.17 | 76.78 | 20.35 | 72.84 | 5.2% | 193 | +1.77 | 0.0% | +0.00 | 1.80 |
| OODD (official) | 90.85 | 97.10 | 67.28 | 32.89 | 54.13 | 38.7% | 512 | -2.46 | 60.0% | -17.09 | 5.60 |
| OODD + T-out (official) | 92.25 | 97.41 | 69.88 | 29.07 | 58.29 | 36.6% | 459 | -1.05 | 46.7% | -11.55 | 3.80 |
| OODD (official, batch 128) | 90.68 | 97.04 | 67.05 | 33.75 | 54.59 | 38.9% | 502 | -2.63 | 60.0% | -16.55 | 6.27 |
| AdaODD* (paper) | 93.97 | 97.43 | 76.25 | 26.16 | 69.50 | - | - | +0.66 | 6.7% | -0.85 | 3.40 |
| WARDEN* (paper) | 94.06 | 97.39 | 76.52 | 24.47 | 68.54 | 0.5% | 27 | +0.76 | 20.0% | -2.75 | 3.20 |
| Age-expiry queue | 67.40 | 87.07 | 56.50 | 75.22 | 47.03 | 69.1% | 502 | -25.91 | 80.0% | -70.71 | 7.07 |

## Same-base comparison: MSP

### all (27 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MSP | 87.96 | 94.41 | 60.27 | 43.45 | 40.91 | - | - | +0.00 | 0.0% | +0.00 | 3.19 |
| **CURE[MSP]** | 90.10 | 95.80 | 65.27 | 37.19 | 52.83 | 5.4% | 113 | +2.14 | 0.0% | -0.00 | 2.04 |
| DCAC-MSP (official) | 69.54 | 87.90 | 39.83 | 69.46 | 18.23 | 50.9% | 291 | -18.42 | 66.7% | -44.57 | 5.26 |
| DCAC-MSP (official, batch 128) | 69.73 | 87.97 | 39.89 | 69.34 | 18.33 | 50.8% | 280 | -18.23 | 66.7% | -42.88 | 5.22 |
| DynProto* (paper) | 89.36 | 95.96 | 64.10 | 46.06 | 52.52 | 50.1% | 404 | +1.39 | 25.9% | -5.26 | 1.96 |
| RTL++ (official) | 77.15 | 87.11 | 46.43 | 87.37 | 24.54 | - | - | -10.81 | 100.0% | -13.09 | 5.41 |
| RTL++ (official, batch 128) | 77.93 | 87.96 | 47.87 | 82.47 | 27.50 | - | - | -10.03 | 100.0% | -13.75 | 4.93 |

Paired comparison (positive = CURE better; FPR95 sign flipped; 95% bootstrap CI over groups; one-sided Wilcoxon).

| Rival | Metric | Diff | 95% CI | Win / loss | p |
|---|---|---|---|---|---|
| DCAC-MSP (official) | AUROC | +20.56 | [+14.50, +26.52] | 22 / 5 | 1.9e-06 |
| DCAC-MSP (official) | FPR95 | +32.28 | [+23.17, +41.12] | 21 / 6 | 4.8e-06 |
| DCAC-MSP (official) | AUPR-Out | +25.44 | [+17.68, +33.53] | 24 / 3 | 8.2e-07 |
| DCAC-MSP (official) | TPR@FPR5 | +34.61 | [+24.25, +45.22] | 23 / 4 | 1.0e-06 |
| DCAC-MSP (official, batch 128) | AUROC | +20.37 | [+14.40, +26.21] | 22 / 5 | 1.9e-06 |
| DCAC-MSP (official, batch 128) | FPR95 | +32.15 | [+23.09, +40.88] | 21 / 6 | 4.8e-06 |
| DCAC-MSP (official, batch 128) | AUPR-Out | +25.38 | [+17.65, +33.43] | 24 / 3 | 8.2e-07 |
| DCAC-MSP (official, batch 128) | TPR@FPR5 | +34.51 | [+24.16, +45.13] | 23 / 4 | 1.0e-06 |
| DynProto* (paper) | AUROC | +0.75 | [+0.03, +1.52] | 15 / 12 | 6.7e-02 |
| DynProto* (paper) | FPR95 | +8.87 | [+3.93, +14.15] | 18 / 9 | 5.2e-03 |
| DynProto* (paper) | AUPR-Out | +1.16 | [-0.63, +2.90] | 18 / 9 | 6.1e-02 |
| DynProto* (paper) | TPR@FPR5 | +0.32 | [-2.86, +3.26] | 14 / 13 | 3.2e-01 |
| RTL++ (official) | AUROC | +12.95 | [+11.93, +14.04] | 27 / 0 | 7.5e-09 |
| RTL++ (official) | FPR95 | +50.18 | [+44.98, +55.31] | 27 / 0 | 7.5e-09 |
| RTL++ (official) | AUPR-Out | +18.84 | [+15.94, +21.92] | 27 / 0 | 7.5e-09 |
| RTL++ (official) | TPR@FPR5 | +28.29 | [+21.65, +35.47] | 27 / 0 | 7.5e-09 |
| RTL++ (official, batch 128) | AUROC | +12.17 | [+11.32, +13.03] | 27 / 0 | 7.5e-09 |
| RTL++ (official, batch 128) | FPR95 | +45.28 | [+39.06, +51.21] | 27 / 0 | 7.5e-09 |
| RTL++ (official, batch 128) | AUPR-Out | +17.40 | [+15.00, +19.99] | 27 / 0 | 7.5e-09 |
| RTL++ (official, batch 128) | TPR@FPR5 | +25.33 | [+19.70, +31.40] | 27 / 0 | 7.5e-09 |

### near (12 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MSP | 85.83 | 93.37 | 56.36 | 51.41 | 36.19 | - | - | +0.00 | 0.0% | +0.00 | 3.25 |
| **CURE[MSP]** | 86.33 | 93.71 | 57.34 | 50.60 | 39.12 | 6.1% | 12 | +0.50 | 0.0% | -0.00 | 2.58 |
| DCAC-MSP (official) | 74.61 | 89.42 | 43.32 | 67.30 | 21.54 | 46.3% | 315 | -11.22 | 50.0% | -30.07 | 4.25 |
| DCAC-MSP (official, batch 128) | 74.56 | 89.37 | 43.26 | 67.40 | 21.60 | 46.3% | 300 | -11.28 | 50.0% | -29.83 | 4.75 |
| DynProto* (paper) | 85.77 | 94.37 | 55.67 | 53.67 | 36.46 | 47.0% | 466 | -0.06 | 41.7% | -5.26 | 2.17 |
| RTL++ (official) | 74.11 | 85.69 | 42.34 | 90.87 | 18.73 | - | - | -11.72 | 100.0% | -12.77 | 5.50 |
| RTL++ (official, batch 128) | 74.40 | 86.48 | 43.08 | 86.41 | 21.20 | - | - | -11.43 | 100.0% | -13.75 | 5.50 |

### far (15 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MSP | 89.67 | 95.24 | 63.39 | 37.08 | 44.68 | - | - | +0.00 | 0.0% | +0.00 | 3.13 |
| **CURE[MSP]** | 93.12 | 97.47 | 71.61 | 26.46 | 63.81 | 5.2% | 193 | +3.45 | 0.0% | -0.00 | 1.60 |
| DCAC-MSP (official) | 65.48 | 86.70 | 37.03 | 71.20 | 15.57 | 54.5% | 272 | -24.19 | 80.0% | -44.57 | 6.07 |
| DCAC-MSP (official, batch 128) | 65.87 | 86.86 | 37.19 | 70.90 | 15.71 | 54.4% | 263 | -23.80 | 80.0% | -42.88 | 5.60 |
| DynProto* (paper) | 92.23 | 97.24 | 70.84 | 39.97 | 65.36 | 52.5% | 355 | +2.56 | 13.3% | -2.97 | 1.80 |
| RTL++ (official) | 79.58 | 88.24 | 49.70 | 84.57 | 29.19 | - | - | -10.09 | 100.0% | -13.09 | 5.33 |
| RTL++ (official, batch 128) | 80.76 | 89.14 | 51.70 | 79.32 | 32.54 | - | - | -8.91 | 100.0% | -11.74 | 4.47 |

## Same-base comparison: Energy

### all (27 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Energy | 87.90 | 93.77 | 64.01 | 53.75 | 50.26 | - | - | +0.00 | 0.0% | +0.00 | 2.67 |
| **CURE[Energy]** | 90.02 | 95.47 | 67.94 | 42.96 | 58.54 | 5.4% | 113 | +2.12 | 0.0% | -0.00 | 1.44 |
| DCAC-Energy (official) | 87.29 | 93.80 | 64.81 | 64.41 | 53.75 | 50.9% | 291 | -0.61 | 40.7% | -4.31 | 2.41 |
| DCAC-Energy (official, batch 128) | 86.43 | 92.82 | 64.31 | 69.34 | 53.22 | 50.8% | 280 | -1.47 | 55.6% | -5.90 | 3.48 |

Paired comparison (positive = CURE better; FPR95 sign flipped; 95% bootstrap CI over groups; one-sided Wilcoxon).

| Rival | Metric | Diff | 95% CI | Win / loss | p |
|---|---|---|---|---|---|
| DCAC-Energy (official) | AUROC | +2.73 | [+1.93, +3.54] | 24 / 3 | 5.2e-07 |
| DCAC-Energy (official) | FPR95 | +21.45 | [+13.76, +29.56] | 22 / 5 | 6.7e-06 |
| DCAC-Energy (official) | AUPR-Out | +3.13 | [+2.05, +4.29] | 25 / 2 | 1.5e-06 |
| DCAC-Energy (official) | TPR@FPR5 | +4.79 | [+2.19, +7.48] | 20 / 7 | 6.9e-04 |
| DCAC-Energy (official, batch 128) | AUROC | +3.59 | [+2.58, +4.57] | 24 / 3 | 4.1e-07 |
| DCAC-Energy (official, batch 128) | FPR95 | +26.38 | [+17.83, +35.26] | 23 / 4 | 2.3e-06 |
| DCAC-Energy (official, batch 128) | AUPR-Out | +3.63 | [+2.51, +4.86] | 25 / 2 | 1.9e-07 |
| DCAC-Energy (official, batch 128) | TPR@FPR5 | +5.32 | [+2.70, +8.09] | 20 / 7 | 4.6e-04 |

### near (12 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Energy | 85.31 | 92.59 | 58.39 | 60.57 | 42.32 | - | - | +0.00 | 0.0% | +0.00 | 2.50 |
| **CURE[Energy]** | 85.87 | 93.13 | 59.23 | 58.60 | 44.34 | 6.1% | 12 | +0.56 | 0.0% | -0.00 | 1.67 |
| DCAC-Energy (official) | 84.56 | 92.67 | 57.42 | 65.19 | 40.76 | 46.3% | 315 | -0.75 | 41.7% | -4.31 | 2.42 |
| DCAC-Energy (official, batch 128) | 83.95 | 91.88 | 57.07 | 69.06 | 40.48 | 46.3% | 300 | -1.36 | 50.0% | -5.45 | 3.42 |

### far (15 groups)

| Method | AUROC | AUPR-In | AUPR-Out | FPR95 | TPR@FPR5 | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Energy | 89.97 | 94.71 | 68.51 | 48.29 | 56.61 | - | - | +0.00 | 0.0% | +0.00 | 2.80 |
| **CURE[Energy]** | 93.34 | 97.34 | 74.91 | 30.45 | 69.90 | 5.2% | 193 | +3.37 | 0.0% | -0.00 | 1.27 |
| DCAC-Energy (official) | 89.47 | 94.71 | 70.72 | 63.79 | 64.15 | 54.5% | 272 | -0.50 | 40.0% | -4.14 | 2.40 |
| DCAC-Energy (official, batch 128) | 88.41 | 93.57 | 70.10 | 69.56 | 63.41 | 54.4% | 263 | -1.56 | 60.0% | -5.90 | 3.53 |
