"""H58a: bản vendor của OODD cho đúng điểm của KNNPostprocessor.conf_postprocess gốc trên cùng đầu vào."""
import ast
import os
from pathlib import Path
from queue import PriorityQueue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cure_ood.methods.testtime._official import oodd_knn as vendored

SRC = Path(os.environ.get("CURE_OODD_REPO", Path(os.environ.get("CURE_LEGACY", ".")) / "baselines_official" / "OODD"))
POST = SRC / "openood" / "postprocessors" / "knn_postprocessor.py"


def upstream_conf_postprocess():
    """Lấy nguyên văn conf_postprocess cùng các hàm phụ từ file gốc (tránh import cả gói openood)."""
    if not POST.exists():
        pytest.skip(f"thiếu repo OODD gốc ({POST}); đặt CURE_OODD_REPO")
    tree = ast.parse(POST.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign))
            and getattr(n, "name", None) in (None, "kth_largest_per_column", "batched_matrix_multiply", "ScoreData")]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KNNPostprocessor")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "conf_postprocess")
    fn.decorator_list = []
    scope = {"np": np, "torch": torch, "PriorityQueue": PriorityQueue, "tqdm": lambda x, **k: x}
    exec(compile(ast.Module(body=[*keep, fn], type_ignores=[]), str(POST), "exec"), scope)
    return scope["conf_postprocess"]


@pytest.mark.parametrize("use_aux", [False, True])
def test_vendored_matches_upstream(use_aux):
    upstream = upstream_conf_postprocess()
    rng = np.random.default_rng(0)
    norm = vendored.normalizer
    act = norm(rng.normal(size=(3000, 64))).astype(np.float32)
    idx = rng.permutation(3000)
    ftest, food = norm(rng.normal(size=(1500, 64))).astype(np.float32), norm(rng.normal(1, 1, (700, 64))).astype(np.float32)
    aux = norm(rng.normal(1, 1, (128, 64))).astype(np.float32) if use_aux else None
    me = SimpleNamespace(activation_log=act, idx=idx, ALPHA=0.5, queue_size=512, K1=10, K2=5, K=10, count=0,
                         aux_feature={"t_out": aux} if use_aux else None)
    s_id, s_ood = upstream(me, food, ftest)
    # Tái tạo đúng phép xáo trộn bên trong hàm gốc rồi đưa luồng đã xáo vào bản vendor.
    all_data = np.concatenate([ftest, food], axis=0)
    label_id = np.concatenate([np.ones(len(ftest)), np.zeros(len(food))])
    np.random.seed(100)
    perm = np.random.permutation(len(all_data))
    s, _ = vendored.conf_postprocess(act, idx, all_data[perm], 10, 5, 0.5, 512, 512, aux, K_prior=10)
    assert np.array_equal(s[label_id[perm] == 1], s_id) and np.array_equal(s[label_id[perm] == 0], s_ood)
