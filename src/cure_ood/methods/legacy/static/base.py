"""Khung chung cho phương pháp tĩnh: thống kê train dùng chung và chọn siêu tham số kiểu APS (OpenOOD v1.5)."""
from __future__ import annotations

import itertools
from functools import cached_property
from typing import ClassVar

import numpy as np
from sklearn.metrics import roc_auc_score

from ....data.benchmark import Benchmark
from ...base import Method, Output


class TrainStats:
    """Thống kê trên tập train (gallery), tính lười; gắn vào benchmark để các phương pháp dùng chung."""

    def __init__(self, bench: Benchmark):
        self.bench = bench

    @cached_property
    def tr(self):
        return self.bench.train_feats

    @cached_property
    def fc(self):
        return self.bench.fc

    @cached_property
    def zt(self):
        # Tính lại từ fc thay vì đọc train_logits: bản gốc làm vậy, giữ đúng từng bit.
        W, b = self.fc
        return self.tr @ W.T + b

    @cached_property
    def mu_g(self):
        return self.tr.mean(0)

    @cached_property
    def n_classes(self) -> int:
        return int(self.bench.train_labels.max()) + 1

    @cached_property
    def class_mu(self):
        lab = self.bench.train_labels
        return np.stack([self.tr[lab == c].mean(0) for c in range(self.n_classes)])

    @cached_property
    def precision(self):
        """Nghịch đảo hiệp phương sai trong lớp dùng chung (tied), như Lee et al. 2018."""
        cen = self.tr - self.class_mu[self.bench.train_labels]
        return np.linalg.pinv(cen.T @ cen / len(self.tr))


def train_stats(bench: Benchmark) -> TrainStats:
    # Gắn vào chính benchmark (ko dùng biến toàn cục): cache sống cùng vòng đời của benchmark.
    return bench.__dict__.setdefault("_static_train_stats", TrainStats(bench))


def aps_select(score_fn, grid: dict, val_id, val_ood, valid=None) -> tuple[dict, float]:
    """Chọn tổ hợp có AUROC (val ID vs val OOD) cao nhất; hòa thì giữ tổ hợp gặp trước, như OpenOOD."""
    best = None
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals))
        if valid is not None and not valid(p):
            continue
        si, so = score_fn(*val_id, **p), score_fn(*val_ood, **p)
        a = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(so))], np.r_[si, so])
        if best is None or a > best[1]:
            best = (p, a)
    return best


class StaticMethod(Method):
    """Phương pháp tĩnh: siêu tham số cho tường minh > APS (nếu bật và benchmark có val_ood) > mặc định."""
    grid: ClassVar[dict] = {}  # lưới APS từ configs/postprocessors của OpenOOD v1.5
    defaults: ClassVar[dict] = {}  # mặc định của bài gốc, dùng khi ko có OOD proxy (vd ImageNet-200)

    def __init__(self, aps: bool = True, **hparams):
        unknown = set(hparams) - set(self.defaults)
        if unknown:
            raise TypeError(f"{type(self).__name__}: siêu tham số ko hợp lệ {sorted(unknown)}")
        self.aps, self.fixed = aps, hparams

    def default_params(self, bench) -> dict:
        return dict(self.defaults)

    def valid(self, p: dict) -> bool:
        return True

    def prepare(self, bench):
        """Tính thống kê train cần cho phương pháp (gọi một lần trong fit)."""

    def ood_score(self, feats, logits, **p) -> np.ndarray:
        raise NotImplementedError

    def fit(self, bench):
        super().fit(bench)
        self.prepare(bench)
        self.params, self.aps_auroc = {**self.default_params(bench), **self.fixed}, None
        free = [k for k in self.grid if k not in self.fixed]
        if self.aps and bench.val_ood and free:
            grid = {k: ([self.fixed[k]] if k in self.fixed else v) for k, v in self.grid.items()}
            val_id = (bench.cal_feats, bench.cal_logits)
            self.params, self.aps_auroc = aps_select(self.ood_score, grid, val_id, bench.val_ood_data(), self.valid)
        return self

    def score(self, stream):
        return Output(self.ood_score(stream.feats, stream.logits, **self.params))
