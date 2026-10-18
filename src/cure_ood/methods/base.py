"""Giao diện chung của mọi phương pháp và registry để tạo phương pháp từ tên trong config."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..data.benchmark import Benchmark, Stream
from ..ops import kth_sim
from ..stats import id_threshold

REGISTRY: dict[str, type[Method]] = {}


@dataclass
class Output:
    scores: np.ndarray  # điểm OOD, lớn = lạ
    memory: list | None = None  # (đầu batch, chỉ số mẫu trong bộ nhớ dùng để chấm) cho DCR; None = ko có bộ nhớ


class Method:
    """Phương pháp OOD: fit chỉ dùng dữ liệu ID ngoài luồng, score ko được đọc stream.y."""
    requires_logits: ClassVar[bool] = False

    def fit(self, bench: Benchmark) -> Method:
        self.bench = bench
        return self

    def score(self, stream: Stream) -> Output:
        raise NotImplementedError


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def build(name: str, **params) -> Method:
    from . import cure, legacy, static, testtime  # noqa: F401  (đăng ký các phương pháp)
    if name not in REGISTRY:
        raise KeyError(f"Phương pháp '{name}' chưa đăng ký. Có: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


@dataclass
class KnnContext:
    b: np.ndarray  # s0 của luồng: cosine tới láng giềng thứ K trong G (lớn = quen)
    cb: np.ndarray  # s0 của tập hiệu chuẩn C
    passes: np.ndarray  # mẫu qua sàng lọc: s0 ≤ τ
    shadow: np.ndarray  # tập đối chứng S = {c ∈ C: s0(c) ≤ τ}


def knn_context(stream: Stream, q: float = 0.2) -> KnnContext:
    """Điểm kNN và sàng lọc dùng chung cho mọi phương pháp test-time; tính một lần mỗi luồng."""
    key = ("knn", q)
    if key not in stream.cache:
        B = stream.bench
        b, cb = kth_sim(stream.x, B.G, B.k), kth_sim(B.C, B.G, B.k)
        tau = id_threshold(cb, q)
        stream.cache[key] = KnnContext(b, cb, b <= tau, B.C[cb <= tau])
    return stream.cache[key]
