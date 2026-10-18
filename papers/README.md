# Literature review

`literature_review.xlsx` is a structured review of the method papers; the PDFs are grouped in folders by topic.


Each spreadsheet entry has the following columns:

| Column | Description |
|---|---|
| No. | Entry number, `<group>.<index>` |
| Paper | Full title |
| Authors | First author, et al. |
| Year / venue | Publication venue and year |
| Source | Link to the official version or arXiv |
| Dataset & setting | Data, benchmarks and experimental setting used in the paper |
| Methodology | How the method works, with its key parameters |
| Results | Main quantitative results reported in the paper, with table or figure references |
| Strengths | What the method does well, as reported by the paper |
| Limitations | Weaknesses stated by the authors or visible from the design |
| Gaps | What the paper leaves open |

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
