# CURE: Dynamic Memory Curation via Retroactive Eviction for Test-Time OOD Detection

Test-time OOD detectors keep a memory of samples seen during deployment and score new inputs against it. Most of them decide at arrival whether a sample enters memory and never revisit that decision, so an in-distribution sample admitted by mistake keeps biasing later decisions (self-poisoning).

CURE separates *scoring* from *remembering*. Every sample is scored with a calibrated static detector and waits in a review queue that is never used for scoring. After a fixed number of later samples, a label-free test (beta-binomial count against an ID reference set, e-BH certification) decides whether the sample joins the curated dynamic memory (CDM) or is evicted retroactively. The memory channel is combined with the static score and recalibrated on held-out ID data; when the CDM is empty, CURE falls back to the static detector. See [docs/method.md](docs/method.md).

This repository contains the CURE implementation, all comparison methods behind one interface, the benchmark runner, feature extraction, and the literature review.

## Install

```bash
git clone https://github.com/3chh/CURE.git && cd CURE
pip install -e ".[dev]"              # core + tests
pip install -e ".[extract,data]"     # optional: feature extraction, Hugging Face download
```

Python ≥ 3.10, PyTorch ≥ 2.1. A GPU is used when available; everything also runs on CPU.

## Data

Methods run on pre-extracted features (one `.npz` per backbone × ID dataset, convention in [docs/adding_a_backbone.md](docs/adding_a_backbone.md)). They are hosted in a private Hugging Face dataset because of the ImageNet / Places / TinyImageNet licences; ask the maintainers for access, then:

```bash
huggingface-cli login
python scripts/download_data.py features      # ~2.1 GB -> data/features/
python scripts/download_data.py checkpoints   # only needed to re-extract features
```

`CURE_FEATURES` and `CURE_CHECKPOINTS` override the default folders.

| Benchmark config | Backbone | ID | near-OOD | far-OOD | Role |
|---|---|---|---|---|---|
| `cifar10` | ResNet18, official OpenOOD v1.5 | CIFAR-10 | CIFAR-100, TinyImageNet | MNIST, SVHN, Textures, Places365 | main |
| `cifar100` | ResNet18, official OpenOOD v1.5 | CIFAR-100 | CIFAR-10, TinyImageNet | MNIST, SVHN, Textures, Places365 | main |
| `imagenet200` | ResNet18, official OpenOOD v1.5 | ImageNet-200 | SSB-hard, NINCO | Textures | main |
| `cifar10_supcon` | ResNet18 trained with SupCon | CIFAR-10 | as `cifar10` | as `cifar10` | robustness |
| `cifar10_dinov2` | DINOv2 ViT-B/14 (frozen, no logits) | CIFAR-10 | as `cifar10` | as `cifar10` | robustness |

A stream mixes the ID test set with one OOD set at OOD ratio π ∈ {5, 20, 50}%, shuffled with a fixed seed, processed in batches of 128. The gallery and calibration sets never enter the stream; labels are used only for evaluation.

## Usage

```bash
cure-ood list                                   # registered methods and benchmarks
cure-ood run quick                              # one stream, ~1 minute
cure-ood run main                               # main table: 3 benchmarks x 15 OOD pairs x 3 π x 3 seeds
cure-ood run main --benchmark cifar100 --method cure oodd    # shard / subset (resumable)
cure-ood report main                            # results/main/report.md + summary.json
cure-ood run seed35 && cure-ood report seed35   # fresh-seed confirmation (seeds 3-5)
cure-ood run ablation && cure-ood report ablation
cure-ood run realtime && cure-ood report realtime   # round 59: batch-by-batch sessions, per-step latency
cure-ood extract cifar10_resnet18 --image-root /path/to/images   # re-extract features
cure-ood train configs/train/cifar10_resnet18_ce.yaml       # train your own backbone (docs/training.md)
cure-ood latency main --extract cifar10_resnet18            # end-to-end timing: image -> backbone -> method
```

Experiments are YAML files in `configs/experiments/`. A method entry is `label: {method: <registered name>, ...hyper-parameters}`; the label defaults to the method name. Several configurations of one method (ablations) are just several entries. To run 4 processes in parallel, shard by `--benchmark` or `--domain`. Results are stored per stream and per method, so an interrupted run resumes where it stopped. On a 16 GB machine run one benchmark per process (CIFAR-100 with all 26 methods peaks at a few GB) and split with `--method` if memory is tight.

| Experiment | Content |
|---|---|
| `main` | main table: every method on `cifar10`, `cifar100`, `imagenet200`, seeds 0-2 |
| `seed35` | `main` unchanged on seeds 3-5 (confirmation, no re-tuning) |
| `ablation` | CURE with one component changed: no review (`certifier: all`), oracle review (`certifier: oracle_labels`, reads labels, upper bound only), no fallback, Fisher instead of Cauchy, e-LOND, window 128/512, capacity 128/2048 |
| `robustness` | `main` methods on the SupCon and DINOv2 backbones |
| `realtime` | round 59: kNN, CURE (e-BH, e-LOND), OODD, DCAC, AdaODD fed one batch of B ∈ {1, 8, 32, 128} at a time; quality, per-step latency, end-to-end latency at 30 / 100 samples/s |
| `round57` | reproduces the round 57 table with the earlier baseline versions |
| `quick` | one stream, smoke test |

`report.md` contains, depending on the `report` section of the experiment file:

- **Main table** (`ranking`): AUROC, FPR95, share of safe groups (AUROC no more than 0.5 below `reference`), mean rank.
- **Same-base comparison** (`families`): all metrics, DCR, memory size, Δ to the base, harm rate, paired tests (bootstrap CI, Wilcoxon) for all / near / far groups, then a breakdown of AUROC / FPR95 by benchmark (all, near, far) and by π.
- **Static methods** (`static`): AUROC, FPR95 and the share of groups where each method has the best AUROC.
- **ID/OOD decisions**: for methods that return a final conformal p-value (CURE), the realised FPR on ID and TPR on OOD when flagging OOD at p ≤ α, for each α in `decision_alphas` (default 0.05, 0.1).
- **Real-time streaming** (`realtime`): per method and batch size B, AUROC, FPR95, harm / safe vs `reference`, p50 / p95 / p99 of the per-batch processing time and end-to-end latency per sample at 30 and 100 samples/s (experiments with `streaming: true`).
- **CDM activity**: for CURE, mean CDM size per batch, share of the stream scored with memory (coverage), and samples certified / evicted per stream, overall, by near/far and by π.

From Python:

```python
from cure_ood.data import Benchmark
from cure_ood.methods import build
from cure_ood.eval.metrics import detection_metrics

bench = Benchmark.load("cifar10")
stream = bench.stream("svhn", pi=0.2, seed=0)
out = build("cure").fit(bench).score(stream)
print(detection_metrics(stream.y, out.scores))
flags = out.decide(alpha=0.05)    # True = OOD; CURE only (out.pvalues), FPR on ID ≈ 0.05 or below
print(out.memory_stats)           # CDM size, coverage, certified / evicted

sess = build("cure", batch=8).fit(bench).session()   # deployment: one batch at a time, state kept in between
step = sess.step(stream.feats[:8])                     # scores and p-values of exactly these 8 samples
```

## Methods

Comparison methods with public code **call the official code**, vendored verbatim with the upstream commit in each file header (`methods/testtime/_official/`, `methods/static/_openood/`). The adapters only feed precomputed features/logits; every change (for example taking an already ordered stream instead of shuffling internally) is marked `ADAPTED` in the vendored file, and `tests/test_official_*.py` check that the vendored code returns exactly what the upstream function returns. Methods without public code are reimplemented component by component from the paper; `docs/reimplementations/<method>.md` maps every paper element (section, equation, algorithm line) to the code and lists the choices the paper leaves open.

| Name | Type | Source |
|---|---|---|
| `cure` | test-time, curated memory (ours); `base: knn \| msp \| energy` | this repo |
| `oodd` (`t_out: true` for the T-out variant) | test-time dictionary | official code, [zxk1212/OODD](https://github.com/zxk1212/OODD) |
| `dcac_msp`, `dcac_energy` | test-time class-aware cache | official code, [wyqstan/DCAC](https://github.com/wyqstan/DCAC) |
| `rtl_plus` | test-time linear re-scoring | official code (`RTL_OOD`), run per batch of the stream |
| `msp`, `maxlogit`, `energy`, `gen`, `mahalanobis`, `vim`, `react`, `nci`, `fdbd` | static, OpenOOD APS | official OpenOOD v1.5 postprocessors |
| `cadref` | static | official code (shipped with DCAC) |
| `knn` | static | cosine to the K-th neighbour (monotone in the kNN-OOD distance) |
| `adaodd`, `warden`, `dynproto` | test-time | reimplemented from the paper (no public code), see `docs/reimplementations/` |
| `age_queue`, `oodd_core` | ablations | this repo |
| `*_legacy`, `*_r53`, `oodd_core` | versions used in experiment rounds 21-57 | kept only to reproduce earlier tables (`configs/experiments/round57.yaml`) |

`cure-ood list` prints the paper and source of every method. To add a method see [docs/adding_a_method.md](docs/adding_a_method.md); to add a backbone or dataset see [docs/adding_a_backbone.md](docs/adding_a_backbone.md).

## Reproducibility

Three kinds of tests:

- `tests/test_official_*.py`: vendored official code (OODD, DCAC, RTL++, OpenOOD postprocessors) returns exactly the upstream result on the same input.
- `tests/test_<method>.py`: unit tests of each reimplemented component against hand-computed values (AdaODD, WARDEN, DynProto).
- `tests/test_golden_*.py`: CURE, CURE-LF and the round 21-57 versions of the baselines reproduce the stored results of the original experiment scripts bit for bit (AUROC with `==` and memory contamination counts); the feature extractors match the original networks and preprocessing.

```bash
pytest                                     # golden tests read tests/golden/ and skip without features
```

Exceptions: Mahalanobis uses an expanded quadratic form that is ~50x faster than the original per-class einsum (AUROC differs by < 1e-5). Bit-exactness assumes the same library versions and GPU; different BLAS thread counts can change RTL++ and DynProto in the last digits.

## Repository layout

```
configs/
  benchmarks/       one YAML per feature file: splits, OOD sets, K
  experiments/      main, seed35, ablation, robustness, round57, quick
  extract/          feature extraction configs
src/cure_ood/
  ops.py            L2 normalisation, k-th neighbour similarity
  stats.py          conformal p-values, Cauchy/Fisher combination, beta-binomial, e-BH, e-LOND
  data/             Benchmark + Stream construction, feature extraction
  backbones/        backbone registry (OpenOOD ResNet18, ResNet-50, DINOv2)
  methods/          base interface, cure.py, memory.py (shared memory host), static/, testtime/
  eval/             metrics, runner, report
  cli.py
scripts/            download_data.py
tests/              unit, golden and equivalence tests
docs/               method, adding a method / backbone
papers/             literature review (xlsx) and PDFs, see papers/README.md
slides/, eda/       slides and the EDA notebook of the OpenOOD benchmark
```

The PDFs in `papers/` are provided for research reference only; copyright remains with the original authors and publishers.
