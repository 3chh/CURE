"""Chỉ mục gallery cho điểm kNN: tìm chính xác (mặc định), thu nhỏ gallery, giảm độ chính xác số, FAISS xấp xỉ."""
from __future__ import annotations

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
