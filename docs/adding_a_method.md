# Adding a method

Every detector implements one small interface (`src/cure_ood/methods/base.py`):

```python
from cure_ood.methods.base import Method, Output, knn_context, register

@register("my_method")                 # name used in experiment configs
class MyMethod(Method):
    paper = "Author et al., Title, Venue Year"
    code = "https://github.com/..."      # None if reimplemented from the paper
    requires_logits = False              # True -> skipped on feature-only benchmarks (e.g. DINOv2)

    def __init__(self, alpha=0.1):       # hyper-parameters come from the YAML config
        self.alpha = alpha

    def fit(self, bench):                # ID data only: bench.G, bench.C, bench.train_feats/logits/labels, bench.fc
        self.bench = bench
        return self

    def score(self, stream):             # stream.x (L2 features), stream.feats, stream.logits; never stream.y
        ctx = knn_context(stream)        # shared kNN score s0, screening mask and reference set S
        scores = -ctx.b                  # larger = more OOD
        return Output(scores, memory=None)   # memory: [(batch_start, indices_in_memory), ...] to report DCR
```

Rules:

- `fit` may only use in-distribution data outside the stream. `score` must be causal in the sense of the protocol (batch `t` may use samples of batches `≤ t`) and must not read `stream.y`.
- Return OOD-positive scores (larger = more likely OOD). AUROC, AUPR, FPR95 and TPR@FPR5 are computed by the runner.
- Expensive quantities shared by several methods go into `stream.cache` (per stream) or are computed once in `fit` (per benchmark).
- Import the new module in `methods/static/__init__.py` or `methods/testtime/__init__.py` so that it registers itself.

Then add it to an experiment config:

```yaml
methods:
  my_method: {alpha: 0.1}
  my_method_strong: {method: my_method, alpha: 0.5}   # several configurations of one method
```

and run `cure-ood run <experiment> --method my_method my_method_strong` followed by `cure-ood report <experiment>`.

If you port a published method, add a golden test (see `tests/test_golden_*.py`) or at least a sanity test, and state in the docstring every deviation from the official code.
