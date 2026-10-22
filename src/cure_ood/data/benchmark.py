"""Benchmark: file đặc trưng + cách tách gallery/hiệu chuẩn và dựng luồng (quy ước npz: docs/adding_a_backbone.md)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import yaml

from ..ops import DEVICE, l2
from . import hub

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "benchmarks"


def data_root() -> Path:
    """Thư mục chứa file đặc trưng; đổi bằng biến môi trường CURE_FEATURES."""
    return Path(os.environ.get("CURE_FEATURES", Path(__file__).resolve().parents[3] / "data" / "features"))


def _config_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    return path if path.suffix else CONFIG_DIR / f"{name_or_path}.yaml"


@dataclass
class Benchmark:
    name: str
    features: str
    k: int | str  # "aps": chọn bằng APS trên (id_val, val_ood) cho đúng chỉ mục `index`
    near: list[str]
    far: list[str]
    gallery: str = "id_train"
    calibration: str = "id_val"
    id_test: str = "id_test"
    val_ood: str | None = None  # OOD proxy cho APS (chọn siêu tham số tĩnh)
    split: dict | None = None  # giao thức tách khi bộ ID ko có tập train (ImageNet-200)
    stream_salt: list[int] = field(default_factory=list)
    index: dict | None = None  # chỉ mục gallery của điểm kNN (data/gallery.py); None = tìm chính xác
    t_out: str | None = None  # nhóm danh sách T-out của OODD (configs/oodd_t_out.json); None = theo tên benchmark
    backbone: str = ""
    description: str = ""

    @classmethod
    def load(cls, spec: str | dict) -> Benchmark:
        """Tên/đường dẫn file cấu hình, hoặc biến thể {name, base, ...trường ghi đè} khai báo trong file thí nghiệm."""
        if isinstance(spec, dict):
            spec = dict(spec)
            cfg = yaml.safe_load(_config_path(spec.pop("base")).read_text(encoding="utf-8"))
            return cls(**{**cfg, **spec})
        return cls(**yaml.safe_load(_config_path(spec).read_text(encoding="utf-8")))

    @property
    def domains(self) -> list[str]:
        """Thứ tự này vào hạt giống RNG của luồng; đổi thứ tự là đổi luồng."""
        return [*self.near, *self.far]

    @cached_property
    def npz(self):
        local = data_root() / self.features
        path = hub.feature_file(self.features, local)
        if path is None:
            raise FileNotFoundError(f"thiếu {local}: chạy `py scripts/download_data.py` hoặc bật CURE_HF_AUTO")
        return np.load(path)

    def array(self, key: str):
        return self.npz[key] if key in self.npz.files else None

    @cached_property
    def has_logits(self) -> bool:
        return "fc_weight" in self.npz.files

    @cached_property
    def _split(self):
        """Tách theo lớp kiểu ImageNet-200: nửa A → gallery + hiệu chuẩn, nửa B → dữ liệu quen của luồng."""
        sp = self.split
        lab = self.npz[f"{sp['source']}_labels"]
        rng = np.random.default_rng(sp["seed"])
        A, B = [], []
        for c in np.unique(lab):
            i = rng.permutation(np.flatnonzero(lab == c))
            A += list(i[: len(i) // 2])
            B += list(i[len(i) // 2:])
        A = rng.permutation(np.array(A))
        cut = int(sp["gallery_frac"] * len(A))
        return A[:cut], A[cut:], np.array(B)

    def _part(self, role: str, kind: str):
        """Mảng `kind` (feats/logits/labels) của vai trò gallery | calibration | id_test."""
        if self.split is None:
            src = {"gallery": self.gallery, "calibration": self.calibration, "id_test": self.id_test}[role]
            return self.array(f"{src}_{kind}")
        idx = dict(zip(("gallery", "calibration", "id_test"), self._split))[role]
        a = self.array(f"{self.split['source']}_{kind}")
        return None if a is None else a[idx]

    # Dữ liệu ID dùng để "fit" các phương pháp (ko bao giờ nằm trong luồng).
    @cached_property
    def train_feats(self):
        return self._part("gallery", "feats")

    @cached_property
    def train_logits(self):
        return self._part("gallery", "logits")

    @cached_property
    def train_labels(self):
        return self._part("gallery", "labels")

    @cached_property
    def G(self) -> np.ndarray:
        """Gallery đã chuẩn hóa L2."""
        return l2(self.train_feats)

    @cached_property
    def G_device(self) -> torch.Tensor:
        """Gallery trên DEVICE, tải một lần cho mọi phiên trực tuyến."""
        return torch.as_tensor(self.G, device=DEVICE)

    @cached_property
    def gallery_index(self):
        from .gallery import build_index
        if not self.index:  # dùng chung G_device với các phương pháp khác
            from .gallery import ExactIndex
            return ExactIndex(self.G, g=self.G_device)
        return build_index(self.G, self.index, self.train_labels)

    @cached_property
    def k_knn(self) -> int:
        """K của điểm kNN; với k: aps thì chọn trên lưới OpenOOD, bỏ các K lớn hơn cỡ gallery."""
        if self.k != "aps":
            return int(self.k)
        from ..methods.static.knn import K_GRID, aps_select
        idx = self.gallery_index
        p, _ = aps_select(lambda f, z, K: -idx.kth_sim(l2(f), K), {"K": K_GRID}, (self.cal_feats, None),
                          self.val_ood_data(), valid=lambda p: p["K"] <= idx.size)
        return p["K"]

    def knn(self, q: np.ndarray, k: int | None = None, chunk: int = 2048) -> np.ndarray:
        """Cosine tới láng giềng thứ K trong gallery (q đã chuẩn hóa L2), qua chỉ mục của benchmark."""
        return self.gallery_index.kth_sim(q, k or self.k_knn, chunk)

    @cached_property
    def cal_feats(self):
        return self._part("calibration", "feats")

    @cached_property
    def cal_logits(self):
        return self._part("calibration", "logits")

    @cached_property
    def C(self) -> np.ndarray:
        """Tập hiệu chuẩn đã chuẩn hóa L2."""
        return l2(self.cal_feats)

    @cached_property
    def test_feats(self):
        """Mọi mẫu ID có thể vào luồng (chưa rút mẫu, chưa chuẩn hóa)."""
        return self._part("id_test", "feats")

    @property
    def fc(self):
        return (self.npz["fc_weight"], self.npz["fc_bias"]) if self.has_logits else None

    def val_ood_data(self):
        return self.array(f"{self.val_ood}_feats"), self.array(f"{self.val_ood}_logits")

    def stream(self, domain: str, pi: float, seed: int) -> Stream:
        """Luồng = mẫu ID của tập test + n_OOD mẫu rút từ `domain`, xáo trộn; thiếu OOD thì bớt ID để giữ π."""
        pool = self._split[2] if self.split else np.arange(len(self.npz[f"{self.id_test}_labels"]))
        n_avail = len(self.npz[f"{domain}_labels"])
        rng = np.random.default_rng([seed, int(pi * 1000), self.domains.index(domain), *self.stream_salt])
        need = round(pi / (1 - pi) * len(pool))
        n_ood = min(need, n_avail)
        n_id = len(pool) if n_ood == need else round(n_ood * (1 - pi) / pi)
        ids, oods = rng.choice(pool, n_id, replace=False), rng.choice(n_avail, n_ood, replace=False)
        perm = rng.permutation(n_id + n_ood)
        return Stream(self, domain, pi, seed, ids, oods, perm)


@dataclass
class Stream:
    bench: Benchmark
    domain: str
    pi: float
    seed: int
    id_idx: np.ndarray
    ood_idx: np.ndarray
    perm: np.ndarray
    cache: dict = field(default_factory=dict)  # đại lượng dùng chung giữa các phương pháp (vd điểm kNN)

    def _gather(self, kind: str):
        src = self.bench.split["source"] if self.bench.split else self.bench.id_test
        a_id, a_ood = self.bench.array(f"{src}_{kind}"), self.bench.array(f"{self.domain}_{kind}")
        if a_id is None or a_ood is None:
            return None
        return np.r_[a_id[self.id_idx], a_ood[self.ood_idx]][self.perm]

    @cached_property
    def y(self) -> np.ndarray:
        """Nhãn OOD (1 = lạ); chỉ dùng để đánh giá."""
        return np.r_[np.zeros(len(self.id_idx), np.uint8), np.ones(len(self.ood_idx), np.uint8)][self.perm]

    @cached_property
    def feats(self):
        return self._gather("feats")

    @cached_property
    def x(self) -> np.ndarray:
        return l2(self.feats)

    @cached_property
    def logits(self):
        return self._gather("logits")

    @property
    def near(self) -> bool:
        return self.domain in self.bench.near

    def __len__(self):
        return len(self.perm)
