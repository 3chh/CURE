# realtime

135 streams. Metrics in %, averaged over seeds per group (benchmark, OOD set, π), then over groups. FPR95 lower is better; all others higher is better.

## Real-time streaming (45 groups)

Each stream is fed to the method one batch of B samples at a time. Harm / safe: AUROC more than 0.5 below / no more than 0.5 below kNN (offline). Step: processing time of one batch on features (CUDA-synchronised), percentiles per stream then averaged over groups; worst = largest per-group p99. End-to-end per sample at a uniform arrival rate = waiting for the batch to fill + waiting for the previous batch + processing (backbone not included).

| Method | B | AUROC | FPR95 | Harm | Safe | Step p50 (ms) | p95 | p99 | Worst p99 | E2E @30/s mean / p99 (ms) | E2E @100/s mean / p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kNN | 1 | 86.19 | 43.2 | 0.0% | 100.0% | 0.86 | 1.27 | 1.78 | 4.38 | 1 / 2 | 1 / 2 |
| kNN | 8 | 86.19 | 43.2 | 0.0% | 100.0% | 0.94 | 1.35 | 1.75 | 3.03 | 118 / 235 | 36 / 71 |
| kNN | 32 | 86.19 | 43.2 | 0.0% | 100.0% | 1.19 | 1.62 | 2.13 | 8.94 | 518 / 1035 | 156 / 311 |
| kNN | 128 | 86.19 | 43.2 | 0.0% | 100.0% | 2.79 | 3.42 | 3.90 | 12.77 | 2117 / 4203 | 637 / 1263 |
| **CURE** e-BH | 1 | 90.72 | 32.6 | 8.9% | 91.1% | 3.60 | 6.49 | 8.76 | 20.07 | 4 / 10 | 54 / 296 |
| **CURE** e-BH | 8 | 90.29 | 33.6 | 13.3% | 86.7% | 4.33 | 7.24 | 9.28 | 24.50 | 121 / 240 | 40 / 77 |
| **CURE** e-BH | 32 | 90.08 | 34.2 | 4.4% | 95.6% | 4.88 | 7.63 | 9.97 | 29.47 | 522 / 1039 | 160 / 315 |
| **CURE** e-BH | 128 | 89.96 | 34.5 | 0.0% | 100.0% | 8.11 | 11.36 | 13.53 | 24.92 | 2123 / 4209 | 643 / 1269 |
| **CURE** e-LOND | 1 | 88.70 | 36.6 | 0.0% | 100.0% | 2.88 | 5.37 | 7.18 | 18.05 | 3 / 8 | 6 / 64 |
| **CURE** e-LOND | 8 | 88.70 | 36.6 | 0.0% | 100.0% | 3.75 | 6.32 | 7.98 | 15.46 | 121 / 239 | 39 / 76 |
| **CURE** e-LOND | 32 | 88.70 | 36.6 | 0.0% | 100.0% | 4.58 | 7.17 | 8.89 | 16.66 | 521 / 1038 | 160 / 315 |
| **CURE** e-LOND | 128 | 88.78 | 36.6 | 0.0% | 100.0% | 8.03 | 11.25 | 13.96 | 43.91 | 2123 / 4209 | 643 / 1269 |
| OODD (official) | 1 | 86.42 | 42.7 | 51.1% | 48.9% | 2.29 | 3.90 | 5.26 | 9.13 | 2 / 5 | 3 / 8 |
| OODD (official) | 8 | 86.44 | 42.7 | 51.1% | 48.9% | 2.87 | 4.38 | 5.78 | 9.67 | 120 / 237 | 38 / 74 |
| OODD (official) | 32 | 86.50 | 42.5 | 51.1% | 48.9% | 3.18 | 4.61 | 5.97 | 17.98 | 520 / 1037 | 158 / 313 |
| OODD (official) | 128 | 86.58 | 42.1 | 46.7% | 53.3% | 9.30 | 11.11 | 12.63 | 18.20 | 2124 / 4210 | 644 / 1270 |
| DCAC-MSP (official) | 1 | 74.63 | 60.7 | 60.0% | 40.0% | 1.97 | 4.05 | 5.71 | 15.40 | 2 / 6 | 8 / 59 |
| DCAC-MSP (official) | 8 | 74.63 | 60.7 | 60.0% | 40.0% | 2.31 | 4.42 | 6.01 | 20.95 | 119 / 237 | 37 / 74 |
| DCAC-MSP (official) | 32 | 74.62 | 60.7 | 60.0% | 40.0% | 3.40 | 6.22 | 7.90 | 24.11 | 520 / 1037 | 159 / 314 |
| DCAC-MSP (official) | 128 | 74.65 | 60.5 | 60.0% | 40.0% | 7.74 | 12.69 | 16.91 | 53.97 | 2123 / 4209 | 643 / 1269 |
| DCAC-Energy (official) | 1 | 81.02 | 71.0 | 86.7% | 13.3% | 2.21 | 4.55 | 6.25 | 15.90 | 2 / 6 | 8 / 76 |
| DCAC-Energy (official) | 8 | 81.03 | 70.9 | 86.7% | 13.3% | 2.46 | 4.73 | 6.18 | 15.62 | 119 / 238 | 38 / 74 |
| DCAC-Energy (official) | 32 | 81.07 | 70.7 | 86.7% | 13.3% | 3.54 | 6.67 | 8.43 | 21.29 | 520 / 1037 | 159 / 314 |
| DCAC-Energy (official) | 128 | 81.24 | 69.9 | 86.7% | 13.3% | 7.98 | 13.10 | 17.30 | 37.60 | 2123 / 4209 | 643 / 1269 |
| AdaODD* (paper) | 1 | 86.29 | 45.3 | 22.2% | 77.8% | 6.05 | 7.50 | 8.98 | 13.42 | 6 / 9 | 9 / 67 |
| AdaODD* (paper) | 8 | 86.29 | 45.3 | 22.2% | 77.8% | 6.19 | 7.56 | 8.79 | 12.98 | 123 / 241 | 41 / 77 |
| AdaODD* (paper) | 32 | 86.28 | 45.3 | 22.2% | 77.8% | 6.91 | 8.51 | 9.76 | 16.21 | 523 / 1041 | 162 / 317 |
| AdaODD* (paper) | 128 | 86.27 | 45.3 | 22.2% | 77.8% | 8.59 | 9.90 | 10.69 | 15.41 | 2123 / 4209 | 643 / 1269 |

## ID/OOD decisions (flag OOD when p ≤ α)

Realised FPR on ID and TPR on OOD in %. p is conformal against the calibration set, so FPR should stay near α or below when the ID part of the stream is exchangeable with it.

| Method | Groups | α | FPR (mean) | FPR (worst group) | TPR |
|---|---|---|---|---|---|
| **CURE** e-LOND | 45 | 0.05 | 4.95 | 5.54 | 51.19 |
| **CURE** e-LOND | 45 | 0.1 | 10.22 | 11.66 | 65.96 |
| **CURE** e-BH | 45 | 0.05 | 5.02 | 6.01 | 57.20 |
| **CURE** e-BH | 45 | 0.1 | 10.17 | 11.67 | 70.20 |
| **CURE** e-BH | 45 | 0.05 | 5.02 | 5.96 | 56.46 |
| **CURE** e-BH | 45 | 0.1 | 10.15 | 11.80 | 69.75 |
| **CURE** e-BH | 45 | 0.05 | 5.09 | 6.03 | 58.53 |
| **CURE** e-BH | 45 | 0.1 | 10.14 | 11.66 | 70.87 |
| **CURE** e-LOND | 45 | 0.05 | 4.96 | 5.55 | 51.44 |
| **CURE** e-LOND | 45 | 0.1 | 10.22 | 11.71 | 66.20 |
| **CURE** e-LOND | 45 | 0.05 | 4.96 | 5.61 | 51.17 |
| **CURE** e-LOND | 45 | 0.1 | 10.23 | 11.66 | 65.95 |
| **CURE** e-LOND | 45 | 0.05 | 4.95 | 5.57 | 51.19 |
| **CURE** e-LOND | 45 | 0.1 | 10.22 | 11.66 | 65.96 |
| **CURE** e-BH | 45 | 0.05 | 5.14 | 6.09 | 59.22 |
| **CURE** e-BH | 45 | 0.1 | 10.19 | 12.04 | 71.78 |

## CDM activity

CDM: mean CDM size per batch. Coverage: share of the stream scored with a non-empty memory. Certified / evicted: samples per stream that left the review queue into CDM / out of the system.

| Method | Part | Groups | CDM | Coverage | Certified | Evicted |
|---|---|---|---|---|---|---|
| **CURE** e-LOND | all | 45 | 86 | 31.0% | 773 | 10151 |
| **CURE** e-LOND | near | 18 | 0 | 1.8% | 0 | 10136 |
| **CURE** e-LOND | far | 27 | 143 | 50.6% | 1288 | 10161 |
| **CURE** e-LOND | π = 0.05 | 15 | 0 | 0.0% | 0 | 8291 |
| **CURE** e-LOND | π = 0.2 | 15 | 76 | 32.3% | 323 | 9570 |
| **CURE** e-LOND | π = 0.5 | 15 | 182 | 60.9% | 1997 | 12594 |
| **CURE** e-BH | all | 45 | 136 | 60.7% | 1134 | 9782 |
| **CURE** e-BH | near | 18 | 14 | 38.8% | 26 | 10101 |
| **CURE** e-BH | far | 27 | 218 | 75.3% | 1872 | 9569 |
| **CURE** e-BH | π = 0.05 | 15 | 3 | 23.0% | 6 | 8283 |
| **CURE** e-BH | π = 0.2 | 15 | 119 | 64.2% | 471 | 9406 |
| **CURE** e-BH | π = 0.5 | 15 | 287 | 94.9% | 2925 | 11657 |
| **CURE** e-BH | all | 45 | 129 | 52.3% | 1102 | 9753 |
| **CURE** e-BH | near | 18 | 8 | 23.6% | 14 | 10042 |
| **CURE** e-BH | far | 27 | 210 | 71.4% | 1828 | 9561 |
| **CURE** e-BH | π = 0.05 | 15 | 1 | 14.7% | 2 | 8268 |
| **CURE** e-BH | π = 0.2 | 15 | 110 | 59.0% | 444 | 9336 |
| **CURE** e-BH | π = 0.5 | 15 | 277 | 83.3% | 2861 | 11655 |
| **CURE** e-BH | all | 45 | 153 | 74.9% | 1207 | 9718 |
| **CURE** e-BH | near | 18 | 32 | 65.5% | 62 | 10075 |
| **CURE** e-BH | far | 27 | 234 | 81.3% | 1970 | 9480 |
| **CURE** e-BH | π = 0.05 | 15 | 7 | 40.6% | 13 | 8277 |
| **CURE** e-BH | π = 0.2 | 15 | 140 | 86.7% | 530 | 9363 |
| **CURE** e-BH | π = 0.5 | 15 | 311 | 97.5% | 3077 | 11514 |
| **CURE** e-LOND | all | 45 | 86 | 32.3% | 772 | 10084 |
| **CURE** e-LOND | near | 18 | 0 | 3.5% | 0 | 10056 |
| **CURE** e-LOND | far | 27 | 144 | 51.5% | 1286 | 10102 |
| **CURE** e-LOND | π = 0.05 | 15 | 0 | 0.0% | 0 | 8270 |
| **CURE** e-LOND | π = 0.2 | 15 | 76 | 34.2% | 321 | 9460 |
| **CURE** e-LOND | π = 0.5 | 15 | 183 | 62.7% | 1994 | 12522 |
| **CURE** e-LOND | all | 45 | 86 | 31.0% | 773 | 10143 |
| **CURE** e-LOND | near | 18 | 0 | 1.8% | 0 | 10127 |
| **CURE** e-LOND | far | 27 | 143 | 50.5% | 1288 | 10154 |
| **CURE** e-LOND | π = 0.05 | 15 | 0 | 0.0% | 0 | 8289 |
| **CURE** e-LOND | π = 0.2 | 15 | 76 | 32.2% | 322 | 9554 |
| **CURE** e-LOND | π = 0.5 | 15 | 182 | 60.8% | 1996 | 12586 |
| **CURE** e-LOND | all | 45 | 86 | 31.1% | 773 | 10154 |
| **CURE** e-LOND | near | 18 | 0 | 1.8% | 0 | 10139 |
| **CURE** e-LOND | far | 27 | 143 | 50.6% | 1289 | 10164 |
| **CURE** e-LOND | π = 0.05 | 15 | 0 | 0.0% | 0 | 8292 |
| **CURE** e-LOND | π = 0.2 | 15 | 76 | 32.3% | 323 | 9571 |
| **CURE** e-LOND | π = 0.5 | 15 | 182 | 60.9% | 1997 | 12600 |
| **CURE** e-BH | all | 45 | 230 | 96.1% | 1574 | 9354 |
| **CURE** e-BH | near | 18 | 135 | 95.1% | 340 | 9799 |
| **CURE** e-BH | far | 27 | 294 | 96.8% | 2396 | 9057 |
| **CURE** e-BH | π = 0.05 | 15 | 45 | 93.6% | 92 | 8200 |
| **CURE** e-BH | π = 0.2 | 15 | 231 | 96.7% | 830 | 9064 |
| **CURE** e-BH | π = 0.5 | 15 | 414 | 98.1% | 3799 | 10798 |
