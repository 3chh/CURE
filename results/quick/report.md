# quick

1 streams. Metrics in %, averaged over seeds per group (benchmark, OOD set, π), then over groups. FPR95 lower is better; all others higher is better.

## Main table (1 groups)

| Method | AUROC | FPR95 | Safe groups | Mean rank |
|---|---|---|---|---|
| knn | 92.69 | 21.6 | 100.0% | 2.00 |
| cure | 97.19 | 13.4 | 100.0% | 1.00 |

Safe group: AUROC no more than 0.5 below knn. Ties in rank are broken by row order.

## ID/OOD decisions (flag OOD when p ≤ α)

Realised FPR on ID and TPR on OOD in %. p is conformal against the calibration set, so FPR should stay near α or below when the ID part of the stream is exchangeable with it.

| Method | Groups | α | FPR (mean) | FPR (worst group) | TPR |
|---|---|---|---|---|---|
| cure | 1 | 0.05 | 4.88 | 4.88 | 83.78 |
| cure | 1 | 0.1 | 9.86 | 9.86 | 92.00 |

## CDM activity

CDM: mean CDM size per batch. Coverage: share of the stream scored with a non-empty memory. Certified / evicted: samples per stream that left the review queue into CDM / out of the system.

| Method | Part | Groups | CDM | Coverage | Certified | Evicted |
|---|---|---|---|---|---|---|
| cure | all | 1 | 247 | 97.7% | 542 | 10339 |
| cure | far | 1 | 247 | 97.7% | 542 | 10339 |
| cure | π = 0.2 | 1 | 247 | 97.7% | 542 | 10339 |
