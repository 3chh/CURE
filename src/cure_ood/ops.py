"""Phép toán dùng chung trên đặc trưng: chuẩn hóa L2 và cosine tới láng giềng thứ k."""
from __future__ import annotations

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def l2(a: np.ndarray) -> np.ndarray:
    return (a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)).astype(np.float32)


def kth_sim(q: np.ndarray, g: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """Cosine tới láng giềng thứ k trong g (lớn = giống); q, g đã chuẩn hóa L2."""
    out = np.empty(len(q), np.float32)
    if len(q) == 0:
        return out
    gt = torch.as_tensor(np.ascontiguousarray(g, dtype=np.float32), device=DEVICE)
    # Giữ khối 2048 như bản gốc: đổi hình dạng phép nhân làm lệch bit trên GPU.
    for s in range(0, len(q), chunk):
        qt = torch.as_tensor(np.ascontiguousarray(q[s:s + chunk], dtype=np.float32), device=DEVICE)
        out[s:s + chunk] = (qt @ gt.T).topk(k, dim=1).values[:, -1].cpu().numpy()
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def logsumexp(z: np.ndarray) -> np.ndarray:
    m = z.max(1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]
