"""Giao diện chung của mọi phương pháp và registry để tạo phương pháp từ tên trong config."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..data.benchmark import Benchmark, Stream
from ..stats import id_threshold

REGISTRY: dict[str, type[Method]] = {}


@dataclass
class Output:
    scores: np.ndarray  # điểm OOD, lớn = lạ
    memory: list | None = None  # (đầu batch, chỉ số mẫu trong bộ nhớ dùng để chấm) cho DCR; None = ko có bộ nhớ
    pvalues: np.ndarray | None = None  # p conformal cuối (hiệu chuẩn trên C); None = phương pháp ko cho p
    memory_stats: dict | None = None  # thống kê hoạt động CDM của luồng (MemoryRun.activity)

    @classmethod
    def concat(cls, outs: list[Output], memory_stats: dict | None = None) -> Output:
        """Ghép kết quả các bước của một phiên thành kết quả cả luồng."""
        pv = None if any(o.pvalues is None for o in outs) else np.concatenate([o.pvalues for o in outs])
        mem = None if all(o.memory is None for o in outs) else [m for o in outs for m in o.memory or []]
        return cls(np.concatenate([o.scores for o in outs]), mem, pv, memory_stats)

    def decide(self, alpha: float) -> np.ndarray:
        """Quyết định ID/OOD: báo lạ khi p ≤ alpha; FPR trên ID ≈ alpha khi mẫu ID hoán đổi được với C."""
        if self.pvalues is None:
            raise ValueError("Phương pháp này ko trả p-value")
        return self.pvalues <= alpha


class Method:
    """Phương pháp OOD: fit chỉ dùng dữ liệu ID ngoài luồng, score ko được đọc stream.y (trừ ablation oracle_labels của CURE)."""
    requires_logits: ClassVar[bool] = False

    def supports(self, bench: Benchmark) -> str | None:
        """Lý do ko chạy được trên benchmark này (vd thiếu logit), None nếu chạy được."""
        return "benchmark ko có logit" if self.requires_logits and not bench.has_logits else None

    def fit(self, bench: Benchmark) -> Method:
        self.bench = bench
        return self

    def score(self, stream: Stream) -> Output:
        raise NotImplementedError

    def session(self, domain: str | None = None) -> Session:
        """Phiên trực tuyến sau fit; domain chỉ dùng cho phương pháp cần dữ liệu phụ theo tập lạ (OODD T-out)."""
        raise NotImplementedError(f"{self.name}: chưa có phiên trực tuyến, dùng score()")


class Session:
    """Trạng thái của phương pháp khi nhận luồng từng batch; step chấm đúng các mẫu vừa đến."""

    def __init__(self):
        self.t = 0  # số mẫu đã nhận = chỉ số trong luồng của mẫu đầu batch kế tiếp

    def step(self, feats: np.ndarray, logits: np.ndarray | None = None) -> Output:
        raise NotImplementedError

    def activity(self) -> dict | None:
        """Thống kê hoạt động bộ nhớ từ đầu phiên (như Output.memory_stats); None nếu ko có."""
        return None


def has_session(method: Method) -> bool:
    return type(method).session is not Method.session


def run_session(method: Method, stream: Stream, batch: int, wrap=None) -> Output:
    """Đưa luồng vào một phiên mới theo từng batch cỡ `batch`; wrap(step) để bọc bước (vd đo thời gian)."""
    sess = method.session(stream.domain)
    step = wrap(sess.step) if wrap else sess.step
    z = stream.logits if method.requires_logits else None
    outs = [step(stream.feats[s:s + batch], None if z is None else z[s:s + batch]) for s in range(0, len(stream), batch)]
    return Output.concat(outs, sess.activity())


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


def knn_reference(bench: Benchmark, q: float = 0.2):
    """Phần chỉ phụ thuộc dữ liệu ID: s0 của C, ngưỡng sàng lọc τ và tập đối chứng S."""
    cb = bench.knn(bench.C)
    tau = id_threshold(cb, q)
    return cb, tau, bench.C[cb <= tau]


def knn_context(stream: Stream, q: float = 0.2) -> KnnContext:
    """Điểm kNN và sàng lọc dùng chung cho mọi phương pháp test-time; tính một lần mỗi luồng."""
    key = ("knn", q)
    if key not in stream.cache:
        B = stream.bench
        b = B.knn(stream.x)
        cb, tau, shadow = knn_reference(B, q)
        stream.cache[key] = KnnContext(b, cb, b <= tau, shadow)
    return stream.cache[key]
