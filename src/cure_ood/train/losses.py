"""Registry hàm mục tiêu; tham số riêng của mục tiêu (đầu chiếu, prototype) nằm trong chính module này."""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

LOSSES: dict[str, Callable[..., Objective]] = {}


class Objective(nn.Module):
    """Giao diện: `forward(model, x, y) -> loss`, x gồm `views` khung nhìn nối theo chiều batch."""

    views = 1
    frozen_backbone = False  # True: chỉ huấn luyện model.fc, thân ở chế độ eval và ko lan truyền gradient


def register_loss(name: str):
    """Decorator đăng ký `cls(feat_dim, num_classes, **options) -> Objective`."""
    def deco(cls):
        LOSSES[name] = cls
        return cls
    return deco


def build_loss(name: str, feat_dim: int, num_classes: int, **options) -> Objective:
    if name not in LOSSES:
        raise KeyError(f"loss '{name}' chưa đăng ký; có: {sorted(LOSSES)}")
    return LOSSES[name](feat_dim, num_classes, **options)


@register_loss("ce")
class CrossEntropy(Objective):
    def __init__(self, feat_dim, num_classes):
        super().__init__()

    def forward(self, model, x, y):
        return F.cross_entropy(model(x), y)


@register_loss("linear_ce")
class LinearProbe(Objective):
    """Cross-entropy trên `model.fc` với thân đóng băng (giai đoạn đánh giá tuyến tính của SupCon)."""

    frozen_backbone = True

    def __init__(self, feat_dim, num_classes):
        super().__init__()

    def forward(self, model, x, y):
        with torch.no_grad():
            _, feats = model(x, return_feature=True)
        return F.cross_entropy(model.fc(feats.detach()), y)


@register_loss("supcon")
class SupCon(Objective):
    """Supervised Contrastive (Khosla et al. 2020), đầu chiếu MLP như SupConResNet của SupContrast."""

    views = 2

    def __init__(self, feat_dim, num_classes, temperature: float = 0.1, base_temperature: float = 0.07,
                 proj_dim: int = 128):
        super().__init__()
        self.temperature, self.base_temperature = temperature, base_temperature
        self.head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, proj_dim))

    def forward(self, model, x, y):
        _, feats = model(x, return_feature=True)
        z = F.normalize(self.head(feats).float(), dim=1)  # fp32: logit chia nhiệt độ 0.1 dễ tràn fp16
        return supcon_loss(z, y, self.temperature, self.base_temperature)


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float, base_temperature: float) -> torch.Tensor:
    """SupConLoss của SupContrast, contrast_mode='all'; z đã chuẩn hóa L2, mỗi mẫu có ít nhất một dương (khung nhìn kia)."""
    logits = z @ z.T / temperature
    logits = logits - logits.max(1, keepdim=True).values.detach()
    not_self = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (y[:, None] == y[None, :]) & not_self
    log_prob = logits - torch.log((logits.exp() * not_self).sum(1, keepdim=True))
    mean_log_prob_pos = (pos * log_prob).sum(1) / pos.sum(1)
    return -(temperature / base_temperature) * mean_log_prob_pos.mean()
