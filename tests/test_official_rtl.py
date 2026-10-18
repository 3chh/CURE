"""H58a: bản vendor của RTL++ cho đúng điểm của RTL_OOD.reranking gốc (RTL_plus.py) trên cùng đầu vào."""
import ast
import os
from pathlib import Path

import numpy as np
import pytest

from cure_ood.methods.testtime._official.rtl import RTL_OOD

SRC = Path(os.environ.get("CURE_RTL_REPO", Path(os.environ.get("CURE_LEGACY", ".")) / "baselines_official" / "RTL"))


def upstream_class():
    """Lấy nguyên văn class RTL_OOD từ RTL_plus.py (ko chạy phần script dựng WideResNet và tải dữ liệu)."""
    path = SRC / "RTL_plus.py"
    if not path.exists():
        pytest.skip(f"thiếu code RTL gốc ({path}); đặt CURE_RTL_REPO")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RTL_OOD")
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.preprocessing import StandardScaler, normalize
    scope = {"np": np, "ElasticNet": ElasticNet, "LinearRegression": LinearRegression,
             "StandardScaler": StandardScaler, "normalize": normalize, "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope["RTL_OOD"]


def test_vendored_matches_upstream():
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(300, 40)))
    y = rng.normal(size=(300, 1))
    a = upstream_class()().reranking(X, y, 0.6, 1e-5)
    b = RTL_OOD().reranking(X, y, 0.6, 1e-5)
    assert np.array_equal(a, b)
