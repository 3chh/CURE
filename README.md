# CURE: Dynamic Memory Curation via Retroactive Eviction for Test-Time OOD Detection

## Problem

Test-time out-of-distribution (OOD) detectors keep a memory of samples observed during deployment and score new inputs against it. Most of these methods decide at arrival time whether a sample enters memory. They make that decision from the sample's own score, and they never revisit it. When an in-distribution sample is admitted by mistake, it stays in memory and biases every later decision. We refer to this failure as self-poisoning. Because of it, a memory-based detector can end up less reliable than a static detector that has no memory.

## Approach

CURE decouples two decisions: how a sample is scored, and whether it is remembered. Incoming samples are scored with a calibrated static detector, and each one is also placed in a quarantine buffer that does not affect scoring. Later, once more of the stream has been observed, CURE re-examines each buffered sample with a label-free statistical test. Samples whose evidence holds up are promoted to long-term memory, and all others are evicted retroactively. Stored entries contribute to scoring only while recent observations still support them. The memory signal is combined with the static score through a calibrated combination. If no supported entry exists, the detector falls back to the static score, so memory adds information without being able to silently degrade the detector.

## Contents

```
papers/                  reviewed papers, one folder per group
literature_review.xlsx   structured review of the method papers (English and Vietnamese sheets)
```

Each spreadsheet entry has the following columns:

| Column | Description |
|---|---|
| No. | Entry number, `<group>.<index>` |
| Paper | Full title |
| Authors | First author, et al. |
| Year / venue | Publication venue and year |
| Source | Link to the official version or arXiv |
| Dataset | Data and benchmarks used in the paper's experiments |
| Method | How the method works |
| Strengths | What the method does well, as reported by the paper |
| Limitations | Weaknesses stated by the authors or visible from the design |
| Gap | What the paper leaves open with respect to CURE |

PDF files are named after their entry number in the spreadsheet. For example, `papers/4_.../4.1_OODD_CVPR2025.pdf` corresponds to entry 4.1.

| # | Group | Papers |
|---|---|---|
| 1 | Static OOD detectors | 10 |
| 2 | VLM-based OOD detection and caches | 4 |
| 3 | Test-time adaptation | 6 |
| 4 | Test-time OOD detection with dynamic memory | 7 |
| 5 | Reinforced retention on streams and buffer purification | 11 |
| 6 | Self-poisoning and defenses | 6 |
| 7 | Statistical testing methods used by CURE | 8 |
| | **Method papers (spreadsheet)** | **52** |
| 8 | Benchmarks and evaluation | 4 |

Group 8 is not part of the method review. It covers the benchmark and OOD data used in the experiments (OpenOOD v1.5, ImageNet-O) and the motivation for the evaluation protocol (base-rate effects, precision-recall versus ROC).

The PDFs are provided for research reference only; copyright remains with the original authors and publishers.
