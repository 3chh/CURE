"""Tự tải từ HF: bản local luôn được ưu tiên, CURE_HF_AUTO=0 thì ko gọi mạng."""
from cure_ood.data import hub


def test_local_file_wins(tmp_path, monkeypatch):
    f = tmp_path / "x.npz"
    f.write_bytes(b"")
    monkeypatch.setattr(hub, "_download", lambda *a: (_ for _ in ()).throw(AssertionError("ko được tải")))
    assert hub.feature_file("x.npz", f) == f


def test_disabled_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("CURE_HF_AUTO", "0")
    assert hub.feature_file("x.npz", tmp_path / "x.npz") is None
    assert hub.checkpoint_file("x.ckpt") is None


def test_missing_downloads_from_features_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("CURE_HF_AUTO", "1")
    monkeypatch.setattr(hub, "_download", lambda repo, kind, name: calls.append((repo, kind, name)) or tmp_path)
    assert hub.feature_file("x.npz", tmp_path / "x.npz") == tmp_path
    assert calls == [(hub.DATA_REPO, "dataset", "features/x.npz")]
