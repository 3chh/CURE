"""Backbone trích đặc trưng: `build_backbone(tên, checkpoint)` → đối tượng có `.transform`, `.forward`, `.fc`."""
from . import builtin  # noqa: F401  (đăng ký các backbone có sẵn)
from .base import Backbone, available_backbones, build_backbone, register_backbone, resolve_checkpoint

__all__ = ["Backbone", "available_backbones", "build_backbone", "register_backbone", "resolve_checkpoint"]
