"""Kiểm thử tính nhân quả tuyệt đối và không có rò rỉ tương lai (Zero Lookahead Bias & Causal Immutability).

Kiểm chứng hai tính chất cốt lõi mà Reviewer/Hội đồng yêu cầu:
1. Tính bất biến của điểm số phát ra tại thời điểm t: Điểm số của mẫu t không bao giờ bị ghi đè hay sửa đổi hồi tố.
2. Tính cách ly nghiêm ngặt của hàng chờ: Tại thời điểm t, CDM tuyệt đối không chứa mẫu nào có tuổi < d=256.
"""
import numpy as np
import pytest
import torch

from cure_ood.data import Benchmark
from cure_ood.methods import build
from cure_ood.methods.base import run_session


def test_scores_are_immutable_and_strictly_causal():
    """Điểm số của các mẫu trong luồng phát ra tại thời điểm t là bất biến;
    dù luồng chạy tiếp thêm hàng nghìn mẫu phía sau, điểm số đã chốt tại thời điểm t
    vẫn trùng khớp 100% từng bit."""
    B = Benchmark.load("cifar10")
    stream = B.stream("svhn", pi=0.2, seed=0)

    cure = build("cure", window=256, batch=128, certifier="ebh")
    fitted = cure.fit(B)

    # Chạy luồng đến mẫu thứ 512
    session_early = fitted.session()
    scores_early = []
    for s in range(0, 512, 128):
        chunk = stream.feats[s : s + 128]
        out_chunk = session_early.step(chunk)
        scores_early.append(out_chunk.scores)
    scores_early = np.concatenate(scores_early)

    # Chạy luồng độc lập tiếp tục đến mẫu thứ 2048
    session_full = fitted.session()
    scores_full = []
    for s in range(0, 2048, 128):
        chunk = stream.feats[s : s + 128]
        out_chunk = session_full.step(chunk)
        scores_full.append(out_chunk.scores)
    scores_full = np.concatenate(scores_full)

    # 512 mẫu đầu tiên phải KHỚP 100% TỪNG BIT giữa hai phiên
    assert np.array_equal(scores_early, scores_full[:512]), "Điểm số không được phép thay đổi hồi tố khi các mẫu tương lai xuất hiện!"


def test_cdm_never_contains_samples_younger_than_window():
    """Tại mọi bước t trong quá trình chạy luồng, CDM chỉ được phép chứa các mẫu
    đã đến cách đây ít nhất window = 256 mẫu.
    Tuổi của mọi mục trong CDM: t - idx >= 256."""
    B = Benchmark.load("cifar10")
    stream = B.stream("svhn", pi=0.5, seed=0)

    window = 256
    cure = build("cure", window=window, batch=128, certifier="ebh")
    session = cure.fit(B).session()

    for s in range(0, len(stream), 128):
        chunk = stream.feats[s : s + 128]
        _ = session.step(chunk)
        current_time = session.t

        if len(session.cdm_idx) > 0:
            # Kiểm tra chỉ số của tất cả các mẫu đang nằm trong CDM
            ages = current_time - session.cdm_idx
            # Tuổi của tất cả các mẫu trong CDM phải >= window
            assert np.all(ages >= window), f"Phát hiện rò rỉ: Có mẫu trong CDM có tuổi < {window} tại bước t={current_time}!"
