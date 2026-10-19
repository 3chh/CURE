"""Vòng 59: chạy phiên trực tuyến từng batch hết luồng phải cho đúng kết quả của score(stream) cùng cỡ batch."""
import numpy as np
import pytest
import torch

from cure_ood.data import Benchmark
from cure_ood.eval.latency import queue_latency
from cure_ood.eval.metrics import detection_metrics
from cure_ood.methods import build
from cure_ood.methods.base import run_session

from .conftest import need_features

STREAMS = [("imagenet200", "textures", 0.05, 0), ("imagenet200", "textures", 0.5, 0), ("cifar10", "svhn", 0.2, 0)]
EXACT = {"knn": {}, "oodd": {}, "oodd_tout": {"method": "oodd", "t_out": True}, "dcac_msp": {}, "dcac_energy": {},
         "adaodd": {}}
BENCHES = {}


def stream_of(name, domain, pi, seed):
    if name not in BENCHES:
        BENCHES[name] = Benchmark.load(name)
        need_features(BENCHES[name])
    return BENCHES[name].stream(domain, pi, seed)


def compare(method, stream):
    """(score(stream), phiên chạy với batch của chính phương pháp), mỗi bên một bộ nhớ đệm luồng riêng."""
    m = method.fit(stream.bench)
    stream.cache.clear()
    ref = m.score(stream)
    stream.cache.clear()
    return ref, run_session(m, stream, m.batch)


def assert_same(a, b):
    assert np.array_equal(a.scores, b.scores)
    assert (a.pvalues is None) == (b.pvalues is None) and (a.pvalues is None or np.array_equal(a.pvalues, b.pvalues))
    assert (a.memory is None) == (b.memory is None)
    if a.memory is not None:
        assert len(a.memory) == len(b.memory)
        assert all(s == t and np.array_equal(x, y) for (s, x), (t, y) in zip(a.memory, b.memory))
    assert a.memory_stats == b.memory_stats


@pytest.mark.parametrize("stream_key", STREAMS)
@pytest.mark.parametrize("label", list(EXACT))
def test_session_matches_score_exactly(label, stream_key):
    spec = dict(EXACT[label])
    stream = stream_of(*stream_key)
    assert_same(*compare(build(spec.pop("method", label), **spec), stream))


@pytest.fixture
def shape_invariant(monkeypatch):
    """Nhân ma trận và chuẩn hóa tính bằng float64 rồi làm tròn về float32: bit ko còn phụ thuộc hình dạng phép nhân."""
    matmul, normalize = torch.Tensor.__matmul__, torch.nn.functional.normalize
    monkeypatch.setattr(torch.Tensor, "__matmul__", lambda a, b: matmul(a.double(), b.double()).to(a.dtype))
    monkeypatch.setattr(torch.nn.functional, "normalize", lambda x, **kw: normalize(x.double(), **kw).to(x.dtype))


CURE_CASES = ([(s, c, 128, "knn") for s in STREAMS for c in ("ebh", "elond")] + [(STREAMS[2], "ebh", 128, "msp")]
              + [(STREAMS[1], c, b, "knn") for b in (1, 8, 32) for c in ("ebh", "elond")])


@pytest.mark.parametrize("stream_key,certifier,batch,base", CURE_CASES)
def test_cure_session_exact_with_shape_invariant_matmul(shape_invariant, stream_key, certifier, batch, base):
    """GPU cho bit khác nhau khi đổi số hàng của phép nhân (score() tính s0 theo khối 2048, phiên theo batch);
    bỏ yếu tố đó thì phiên CURE phải khớp từng bit: logic hàng chờ, hậu kiểm nhân quả, CDM và hiệu chuẩn."""
    stream = stream_of(*stream_key)
    ref, out = compare(build("cure", certifier=certifier, batch=batch, base=base), stream)
    assert ref.memory_stats["certified"] > 0 or stream_key == STREAMS[0]  # luồng π 0,05 ko chứng nhận mẫu nào
    assert_same(ref, out)


@pytest.mark.parametrize("stream_key", STREAMS)
@pytest.mark.parametrize("certifier", ["ebh", "elond"])
def test_cure_session_close_on_gpu(stream_key, certifier):
    """Số học GPU thật: s0 lệch ≤ vài ulp nên chỉ vài p conformal đổi một hạng; quyết định hậu kiểm và CDM vẫn trùng."""
    stream = stream_of(*stream_key)
    ref, out = compare(build("cure", certifier=certifier), stream)
    assert ref.memory_stats == out.memory_stats
    assert all(np.array_equal(x, y) for (_, x), (_, y) in zip(ref.memory, out.memory))
    assert np.mean(ref.pvalues != out.pvalues) <= 0.01
    auroc = [detection_metrics(stream.y, o.scores)["auroc"] for o in (ref, out)]
    assert abs(auroc[0] - auroc[1]) <= 1e-5


def test_queue_latency_hand_computed():
    # 4 mẫu, batch 2, tốc độ 1 mẫu/s, xử lý 0,5 s rồi 3 s: batch 1 xong lúc 1,5; batch 2 sẵn lúc 3, xong lúc 6.
    lat = queue_latency(np.array([0.5, 3.0]), batch=2, n=4, rate=1.0)
    assert np.allclose(lat, [1.5, 0.5, 4.0, 3.0])
    # Xử lý chậm hơn nhịp đến: batch sau phải chờ batch trước xong.
    assert np.allclose(queue_latency(np.array([3.0, 0.5]), batch=1, n=2, rate=1.0), [3.0, 2.5])
