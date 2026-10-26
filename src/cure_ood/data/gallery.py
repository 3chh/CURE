"""Chỉ mục gallery cho điểm kNN: tìm chính xác (mặc định), thu nhỏ gallery, giảm độ chính xác số, FAISS xấp xỉ."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from ..ops import DEVICE, kth_sim, l2


class ExactIndex:
    """Cosine chính xác bằng phép nhân ma trận; giữ đúng số học của kth_sim nên bit-exact với các vòng trước."""

    def __init__(self, G: np.ndarray, dtype: torch.dtype = torch.float32, g: torch.Tensor | None = None):
        self.dtype, self.size = dtype, len(G)
        self.g = (g if g is not None else torch.as_tensor(G, device=DEVICE)).to(dtype)

    def kth_sim(self, q: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
        if self.dtype == torch.float32:
            return kth_sim(q, self.g, k, chunk)
        out = np.empty(len(q), np.float32)
        for s in range(0, len(q), chunk):
            qt = torch.as_tensor(np.ascontiguousarray(q[s:s + chunk]), device=DEVICE).to(self.dtype)
            out[s:s + chunk] = (qt @ self.g.T).topk(k, dim=1).values[:, -1].float().cpu().numpy()
        return out


class FaissIndex:
    """FAISS trên tích vô hướng (= cosine vì đã chuẩn hóa L2); kết quả xếp giảm dần nên lấy cột k-1."""

    def __init__(self, G: np.ndarray, factory: str, ef_search: int | None = None, nprobe: int | None = None):
        import faiss
        self.size = len(G)
        self.index = faiss.index_factory(G.shape[1], factory, faiss.METRIC_INNER_PRODUCT)
        G = np.ascontiguousarray(G, dtype=np.float32)
        if not self.index.is_trained:
            self.index.train(G)
        self.index.add(G)
        if ef_search is not None:
            faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", ef_search)
        if nprobe is not None:
            faiss.ParameterSpace().set_index_parameter(self.index, "nprobe", nprobe)

    def kth_sim(self, q: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
        sims, _ = self.index.search(np.ascontiguousarray(q, dtype=np.float32), k)
        # Chỉ mục xấp xỉ có thể trả thiếu láng giềng (-inf hoặc ~-3.4e38): coi như rất xa.
        return np.nan_to_num(sims[:, k - 1], neginf=-1.0).clip(-1.0, 1.0).astype(np.float32)


def subsample(G: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Lấy ngẫu nhiên đều một phần gallery (như kNN-OOD, Sun et al. 2022)."""
    n = max(1, round(frac * len(G)))
    return G[np.sort(np.random.default_rng(seed).choice(len(G), n, replace=False))]


def class_prototypes(G: np.ndarray, labels: np.ndarray, per_class: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """k-means++ theo từng lớp trên đặc trưng chuẩn hóa; tâm được chuẩn hóa lại để vẫn so bằng cosine."""
    from sklearn.cluster import KMeans
    out = []
    for c in np.unique(labels):
        Xc = G[labels == c]
        km = KMeans(n_clusters=min(per_class, len(Xc)), init="k-means++", n_init=1, max_iter=iters, random_state=seed)
        out.append(km.fit(Xc).cluster_centers_)
    return l2(np.concatenate(out).astype(np.float32))


def build_index(G: np.ndarray, spec: dict | None, labels: np.ndarray | None = None):
    """spec: None | {subsample: f} | {prototypes: m} | {precision: bf16} | {faiss: <factory>, ef_search, nprobe}."""
    spec = dict(spec or {})
    seed = spec.pop("seed", 0)
    if "subsample" in spec:
        G = subsample(G, spec.pop("subsample"), seed)
    if "prototypes" in spec:
        G = class_prototypes(G, labels, spec.pop("prototypes"), seed)
    if "faiss" in spec:
        index = FaissIndex(G, spec.pop("faiss"), spec.pop("ef_search", None), spec.pop("nprobe", None))
    else:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[spec.pop("precision", "fp32")]
        index = ExactIndex(G, dtype)
    if spec:
        raise ValueError(f"tham số chỉ mục ko hiểu: {sorted(spec)}")
    return index


class Shards:
    """Gallery lưu thành nhiều file .npy (vd ImageNet-1k 1,28 triệu x 2048 fp16): đọc theo mảnh, ko nạp hết vào RAM."""

    def __init__(self, feats: list[Path], labels: list[Path] | None = None):
        self.parts = [np.load(p, mmap_mode="r") for p in feats]
        self.sizes = [len(p) for p in self.parts]
        self.offsets = np.cumsum([0] + self.sizes)
        self.size, self.dim = int(self.offsets[-1]), int(self.parts[0].shape[1])
        self._labels = labels

    @cached_property
    def labels(self) -> np.ndarray:
        if not self._labels:
            raise ValueError("gallery theo mảnh ko có file nhãn: cần cho prototypes")
        return np.concatenate([np.load(p) for p in self._labels])

    def chunks(self, size: int = 100_000):
        """Lần lượt trả các khối đã chuẩn hóa L2 (float32), ko bao giờ giữ quá `size` dòng."""
        for part in self.parts:
            for s in range(0, len(part), size):
                yield l2(np.asarray(part[s:s + size], dtype=np.float32))

    def rows(self, idx: np.ndarray) -> np.ndarray:
        """Các dòng theo chỉ số toàn cục (đã chuẩn hóa L2); idx phải tăng dần."""
        out = np.empty((len(idx), self.dim), np.float32)
        at = np.searchsorted(self.offsets, idx, side="right") - 1
        for j, part in enumerate(self.parts):
            m = at == j
            if m.any():
                out[m] = np.asarray(part[idx[m] - self.offsets[j]], dtype=np.float32)
        return l2(out)


def _exact_from_shards(sh: Shards, dtype: torch.dtype) -> ExactIndex:
    """Dựng thẳng tensor trên DEVICE theo khối, nên RAM chỉ giữ một khối thay vì cả gallery."""
    g = torch.empty((sh.size, sh.dim), dtype=dtype, device=DEVICE)
    at = 0
    for c in sh.chunks():
        g[at:at + len(c)] = torch.as_tensor(c, device=DEVICE).to(dtype)
        at += len(c)
    idx = ExactIndex.__new__(ExactIndex)
    idx.dtype, idx.size, idx.g = dtype, sh.size, g
    return idx


def _faiss_from_shards(sh: Shards, factory: str, ef_search, nprobe, train_n: int = 262_144) -> FaissIndex:
    import faiss
    index = faiss.index_factory(sh.dim, factory, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:  # huấn luyện trên mẫu con trải đều, như khuyến nghị của FAISS
        step = max(1, sh.size // train_n)
        index.train(np.ascontiguousarray(sh.rows(np.arange(0, sh.size, step))))
    for c in sh.chunks():
        index.add(np.ascontiguousarray(c))
    if ef_search is not None:
        faiss.ParameterSpace().set_index_parameter(index, "efSearch", ef_search)
    if nprobe is not None:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
    out = FaissIndex.__new__(FaissIndex)
    out.index, out.size = index, sh.size
    return out


def build_index_shards(sh: Shards, spec: dict | None):
    """Như build_index nhưng nguồn là gallery theo mảnh; chỉ giữ một khối trong RAM mỗi lúc."""
    spec = dict(spec or {})
    seed = spec.pop("seed", 0)
    if "subsample" in spec:
        n = max(1, round(spec.pop("subsample") * sh.size))
        idx = np.sort(np.random.default_rng(seed).choice(sh.size, n, replace=False))
        return build_index(sh.rows(idx), spec or None)
    if "prototypes" in spec:
        m, lab = spec.pop("prototypes"), sh.labels
        cent = []
        for c in np.unique(lab):
            Xc = sh.rows(np.flatnonzero(lab == c))
            cent.append(class_prototypes(Xc, np.zeros(len(Xc), int), m, seed))
        return build_index(np.concatenate(cent), spec or None)
    if "faiss" in spec:
        return _faiss_from_shards(sh, spec.pop("faiss"), spec.pop("ef_search", None), spec.pop("nprobe", None))
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[spec.pop("precision", "fp32")]
    if spec:
        raise ValueError(f"tham số chỉ mục ko hiểu: {sorted(spec)}")
    return _exact_from_shards(sh, dtype)
