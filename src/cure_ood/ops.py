"""Phép toán dùng chung trên đặc trưng: chuẩn hóa L2 và cosine tới láng giềng thứ k."""
from __future__ import annotations

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def l2(a: np.ndarray) -> np.ndarray:
    return (a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)).astype(np.float32)


def kth_sim(q: np.ndarray, g, k: int, chunk: int = 2048) -> np.ndarray:
    """Cosine tới láng giềng thứ k trong g (lớn = giống); q, g đã chuẩn hóa L2, g có thể là tensor sẵn trên DEVICE."""
    out = np.empty(len(q), np.float32)
    if len(q) == 0:
        return out
    gt = g if torch.is_tensor(g) else torch.as_tensor(np.ascontiguousarray(g, dtype=np.float32), device=DEVICE)
    # Tự động điều chỉnh chunk khi gallery lớn (CIFAR-10 50k, ImageNet) để tránh OOM trên GPU:
    if DEVICE == "cuda" and len(gt) > 10000 and chunk > 512:
        chunk = 512
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
