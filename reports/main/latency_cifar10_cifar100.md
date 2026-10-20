# Latency: cifar10 / cifar100, π = 0.2

Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU. Method times are amortised over the whole stream (11250 samples); per-batch percentiles need the incremental interface.

## Backbone (image → features)

| Batch | Load+preprocess p50 (ms) | Forward p50 (ms) | Total p95 (ms) | Total p99 (ms) | Per sample (ms) | Throughput (img/s) |
|---|---|---|---|---|---|---|
| 1 | 0.53 | 2.62 | 5.33 | 6.85 | 3.496 | 286 |
| 32 | 10.80 | 10.02 | 25.83 | 36.74 | 0.687 | 1456 |
| 128 | 41.97 | 33.60 | 79.28 | 81.12 | 0.595 | 1680 |

## Methods (features → score) and end-to-end

End-to-end per sample = backbone at the largest batch (0.595 ms) + method. Budget 30/s = 33.3 ms per sample. Budget 100/s = 10.0 ms per sample.

| Method | Fit (s) | Method per sample (ms) | Per batch (ms) | Batch | End-to-end per sample (ms) | Samples/s | GPU peak (MB) |
|---|---|---|---|---|---|---|---|
| energy | 0.08 | 0.002 | 0.3 | 128 | 0.597 | 1674 | 10 |
| fdbd | 0.21 | 0.003 | 0.4 | 128 | 0.598 | 1671 | 10 |
| gen | 0.45 | 0.004 | 0.5 | 128 | 0.599 | 1670 | 10 |
| msp | 0.01 | 0.004 | 0.5 | 128 | 0.599 | 1670 | 10 |
| nci | 0.39 | 0.005 | 0.6 | 128 | 0.600 | 1667 | 11 |
| maxlogit | 0.01 | 0.005 | 0.6 | 128 | 0.600 | 1667 | 10 |
| react | 0.10 | 0.006 | 0.8 | 128 | 0.601 | 1663 | 10 |
| cadref | 0.65 | 0.007 | 0.9 | 128 | 0.603 | 1660 | 9 |
| knn | 0.00 | 0.022 | 2.8 | 128 | 0.617 | 1621 | 510 |
| vim | 1.95 | 0.023 | 2.9 | 128 | 0.618 | 1618 | 10 |
| age_queue | 0.00 | 0.025 | 3.1 | 128 | 0.620 | 1614 | 510 |
| cure_energy | 0.00 | 0.037 | 4.8 | 128 | 0.632 | 1581 | 510 |
| warden | 1.82 | 0.039 | 5.0 | 128 | 0.634 | 1576 | 604 |
| dcac_energy | 0.00 | 0.040 | 20.3 | 512 | 0.635 | 1575 | 15 |
| cure_msp | 0.00 | 0.040 | 5.1 | 128 | 0.635 | 1574 | 510 |
| dcac_msp | 0.01 | 0.045 | 22.8 | 512 | 0.640 | 1563 | 15 |
| dcac_msp_protocol | 0.00 | 0.050 | 6.4 | 128 | 0.645 | 1549 | 13 |
| dcac_energy_protocol | 0.00 | 0.054 | 6.8 | 128 | 0.649 | 1541 | 13 |
| mahalanobis | 1.01 | 0.059 | 7.5 | 128 | 0.654 | 1529 | 10 |
| adaodd | 0.10 | 0.075 | 9.6 | 128 | 0.671 | 1491 | 401 |
| cure | 0.00 | 0.084 | 10.7 | 128 | 0.679 | 1472 | 510 |
| oodd | 0.11 | 0.090 | 45.8 | 512 | 0.685 | 1460 | 83 |
| oodd_tout | 0.11 | 0.096 | 49.1 | 512 | 0.691 | 1447 | 83 |
| oodd_protocol | 0.07 | 0.135 | 17.3 | 128 | 0.730 | 1369 | 70 |
| rtl_plus | 0.00 | 0.829 | 424.1 | 512 | 1.425 | 702 | 9 |
| dynproto | 0.04 | 1.083 | 138.4 | 128 | 1.678 | 596 | 9 |
| rtl_plus_protocol | 0.00 | 1.391 | 177.8 | 128 | 1.986 | 504 | 9 |
