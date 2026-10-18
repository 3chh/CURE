"""Đối chiếu backbone mới với đường trích đặc trưng gốc (code của dự án cũ) trên cùng checkpoint, ko cần ảnh thật.

CURE_LEGACY: thư mục dự án gốc (mặc định hai cấp trên repo). CURE_CHECKPOINTS: các thư mục checkpoint, ngăn bởi
os.pathsep (mặc định checkpoint/ của OODD và models/ trong dự án gốc).
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cure_ood.backbones import Backbone, available_backbones, build_backbone
from cure_ood.data.extract import extract, load_config, load_rows, run

REPO = Path(__file__).resolve().parents[1]
LEGACY = Path(os.environ.get("CURE_LEGACY", REPO.parents[1]))
OODD = LEGACY / "baselines_official" / "OODD"
CKPT_DIRS = os.environ.get("CURE_CHECKPOINTS", os.pathsep.join([str(OODD / "checkpoint"), str(LEGACY / "models")]))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOL = {"rtol": 1e-6, "atol": 1e-6}

needs_legacy = pytest.mark.skipif(not (LEGACY / "scripts" / "probe_raw_checkpoint.py").exists(),
                                  reason="ko thấy dự án gốc (CURE_LEGACY)")


def checkpoint(name: str) -> Path:
    for d in CKPT_DIRS.split(os.pathsep):
        if (Path(d) / name).exists():
            return Path(d) / name
    pytest.skip(f"thiếu checkpoint {name}")


def load_module(path: Path, name: str, argv=None):
    """Nạp script gốc như module; vài script đọc sys.argv ngay lúc import."""
    if str(LEGACY) not in sys.path:
        sys.path.insert(0, str(LEGACY))
    old = sys.argv
    sys.argv = argv or [str(path)]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old


def synthetic_images() -> list[Image.Image]:
    """Ảnh ngẫu nhiên nhiều kích thước và chế độ màu, để thử cả resize lẫn convert("RGB")."""
    rng = np.random.default_rng(0)
    sizes = [(32, 32), (40, 28), (300, 257), (17, 64), (256, 256)]
    ims = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in sizes]
    ims.append(Image.fromarray(rng.integers(0, 256, (90, 120), dtype=np.uint8), mode="L"))
    return ims


@pytest.fixture(autouse=True)
def deterministic():
    old = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old


def legacy_openood(ckpt: Path, mode: str):
    """Mô hình và tiền xử lý theo đúng các nhánh của scripts/extract_openood_v15_features.py."""
    mod = load_module(LEGACY / "scripts" / "extract_openood_v15_features.py", f"legacy_v15_{mode}", ["x", mode])
    if mode == "cifar100":
        net = load_module(OODD / "openood" / "networks" / "resnet18_32x32.py", "legacy_r18_32").ResNet18_32x32(num_classes=100)
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model = net.eval().to(DEVICE)
    else:
        model = mod.load_model(DEVICE)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict({k.replace(".short.", ".shortcut."): v for k, v in state.items()}, strict=True)
        model.eval()

    @torch.no_grad()
    def encode(x):
        logits, feats = model(x.to(DEVICE), return_feature=True)
        return feats.cpu(), logits.cpu()

    return encode, mod.ListSet([]).tf


def legacy_in200(ckpt: Path):
    mod = load_module(LEGACY / "scripts" / "extract_in200_features.py", "legacy_in200")
    net = load_module(OODD / "openood" / "networks" / "resnet18_224x224.py", "legacy_r18_224").ResNet18_224x224(num_classes=200)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net = net.eval().to(DEVICE)

    @torch.no_grad()
    def encode(x):
        logits, feats = net(x.to(DEVICE), return_feature=True)
        return feats.cpu(), logits.cpu()

    return encode, mod.TF


def legacy_resnet50(ckpt: Path):
    import torchvision
    mod = load_module(LEGACY / "archive" / "cleanup_2026-10-07" / "scripts" / "extract_imagenet_resnet50_features.py",
                      "legacy_r50")
    net = torchvision.models.resnet50()
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.to(DEVICE).eval()
    body = torch.nn.Sequential(*list(net.children())[:-1])
    w, b = net.fc.weight.detach().cpu().numpy(), net.fc.bias.detach().cpu().numpy()

    def encode(x):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            f = body(x.to(DEVICE)).flatten(1).float().cpu().numpy()
        return torch.from_numpy(f), torch.from_numpy((f @ w.T + b).astype(np.float32))

    return encode, mod.TF


CASES = {  # id: (tên backbone, checkpoint, options, hàm dựng đường gốc, kích thước ảnh)
    "cifar10_official": ("resnet18_32x32", "resnet18_cifar10.ckpt", {"norm": "cifar10"}, lambda c: legacy_openood(c, "official"), 32),
    "cifar10_supcon": ("resnet18_32x32", "resnet18_supcon.pt", {"norm": "cifar10"}, lambda c: legacy_openood(c, "supcon"), 32),
    "cifar100": ("resnet18_32x32", "resnet18_cifar100.ckpt", {"norm": "cifar100"}, lambda c: legacy_openood(c, "cifar100"), 32),
    "imagenet200": ("resnet18_224x224", "resnet18_imagenet200.ckpt", {}, legacy_in200, 224),
    "imagenet1k_r50": ("resnet50_in1k", "resnet50_imagenet1k.pth", {}, legacy_resnet50, 224),
}


@needs_legacy
@pytest.mark.parametrize("case", CASES)
def test_matches_legacy(case):
    name, ckpt_name, options, legacy, size = CASES[case]
    if name == "resnet50_in1k" and DEVICE.type != "cuda":
        pytest.skip("đường gốc ResNet-50 dùng autocast CUDA")
    ckpt = checkpoint(ckpt_name)
    old_encode, old_tf = legacy(ckpt)
    bb = build_backbone(name, ckpt, DEVICE, **options)

    ims = [im.convert("RGB") for im in synthetic_images()]
    x_old = torch.stack([old_tf(im) for im in ims])
    x_new = torch.stack([bb.transform(im) for im in ims])
    assert torch.equal(x_old, x_new), "tiền xử lý lệch"

    x_rand = torch.randn(4, 3, size, size, generator=torch.Generator().manual_seed(0))
    for x in (x_new, x_rand):
        f_old, lo_old = old_encode(x)
        f_new, lo_new = bb.forward(x)
        torch.testing.assert_close(f_new, f_old, **TOL)
        torch.testing.assert_close(lo_new, lo_old, **TOL)

    w, b = bb.fc
    assert w.dtype == np.float32 and w.shape[0] == lo_new.shape[1] and b.shape == (w.shape[0],)


@needs_legacy
def test_dinov2_loads_from_cache():
    hub = Path(torch.hub.get_dir())
    if not ((hub / "facebookresearch_dinov2_main").exists() and (hub / "checkpoints" / "dinov2_vitb14_pretrain.pth").exists()):
        pytest.skip("DINOv2 chưa có trong cache torch.hub; ko tải trong lúc test")
    bb = build_backbone("dinov2_vitb14", device=DEVICE)
    old_tf = load_module(LEGACY / "scripts" / "extract_cifar10_dino_v15.py", "legacy_dino").TF
    for im in synthetic_images():
        assert torch.equal(old_tf(im.convert("RGB")), bb.transform(im.convert("RGB")))
    feats, logits = bb.forward(torch.randn(2, 3, 224, 224))
    assert feats.shape == (2, 768) and feats.dtype == torch.float32 and logits is None and torch.isfinite(feats).all()
    assert bb.fc is None


def test_registry():
    assert {"resnet18_32x32", "resnet18_224x224", "resnet50_in1k", "dinov2_vitb14"} <= set(available_backbones())
    with pytest.raises(KeyError):
        build_backbone("khong_co")


def write_images(root: Path, rels: list[str]):
    ims = synthetic_images()
    for i, rel in enumerate(rels):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        ims[i % len(ims)].save(root / rel)


def test_load_rows(tmp_path):
    """Bỏ trùng, lấy mẫu con theo công thức gốc, gốc ảnh theo tiền tố, bỏ ảnh thiếu."""
    rels = [f"a/{i}.png" for i in range(20)] + ["texture/t.png"]
    write_images(tmp_path / "main", rels[:-1])
    write_images(tmp_path / "classic", rels[-1:])
    lst = tmp_path / "list.txt"
    lst.write_text("".join(f"{r} {i % 3}\n" for i, r in enumerate(rels + rels[:5])) + "missing.png 0\n")

    with pytest.raises(FileNotFoundError):
        load_rows(lst, tmp_path / "main", {"texture/": tmp_path / "classic"})
    rows = load_rows(lst, tmp_path / "main", {"texture/": tmp_path / "classic"}, dedup=True, skip_missing=True)
    assert len(rows) == 21 and rows[-1][0] == tmp_path / "classic" / "texture/t.png"

    sub = load_rows(lst, tmp_path / "main", {"texture/": tmp_path / "classic"}, subsample=7, seed=0, dedup=True,
                    skip_missing=True)
    raw = [ln.split() for ln in lst.read_text().splitlines()][:21] + [["missing.png", "0"]]
    expect = [raw[i] for i in np.random.default_rng(0).permutation(len(raw))[:7] if raw[i][0] != "missing.png"]
    assert [(p.name, lab) for p, lab in sub] == [(Path(r).name, int(lab)) for r, lab in expect]


def test_extract_npz_convention(tmp_path):
    rels = [f"img/{i}.png" for i in range(6)]
    write_images(tmp_path, rels)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)).eval()
    fc = torch.nn.Linear(4, 2)
    from torchvision import transforms
    bb = Backbone("toy", net, transforms.Compose([transforms.Resize(8), transforms.CenterCrop(8), transforms.ToTensor()]),
                  lambda x: (net(x), fc(net(x))), torch.device("cpu"),
                  (fc.weight.detach().numpy(), fc.bias.detach().numpy()))
    rows = [(tmp_path / r, i % 2) for i, r in enumerate(rels)]
    out = tmp_path / "f.npz"
    extract(bb, {"id_test": rows, "ood": [(p, -1) for p, _ in rows[:3]]}, out, batch_size=4)
    z = np.load(out)
    assert set(z.files) == {"fc_weight", "fc_bias", "id_test_feats", "id_test_logits", "id_test_labels",
                            "ood_feats", "ood_logits", "ood_labels"}
    assert z["id_test_feats"].dtype == np.float32 and z["id_test_logits"].dtype == np.float32
    assert z["id_test_labels"].dtype == np.int64 and z["ood_labels"].tolist() == [-1] * 3
    with torch.no_grad():
        direct = net(torch.stack([bb.transform(Image.open(p).convert("RGB")) for p, _ in rows])).numpy()
    np.testing.assert_allclose(z["id_test_feats"], direct, rtol=1e-6, atol=1e-6)


@needs_legacy
def test_run_matches_legacy_on_files(tmp_path, monkeypatch):
    """Toàn bộ đường YAML → npz trên ảnh tổng hợp ghi ra đĩa, so với mô hình + tiền xử lý gốc."""
    ckpt = checkpoint("resnet18_cifar10.ckpt")
    rels = [f"cifar10/test/{i}.png" for i in range(6)]
    write_images(tmp_path / "images" / "images_classic", rels)
    (tmp_path / "lists" / "cifar10").mkdir(parents=True)
    (tmp_path / "lists" / "cifar10" / "mini.txt").write_text("".join(f"{r} {i}\n" for i, r in enumerate(rels)))
    monkeypatch.setenv("CURE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CURE_IMGLISTS", str(tmp_path / "lists"))
    monkeypatch.setenv("CURE_CHECKPOINTS", str(ckpt.parent))
    cfg = {**load_config("cifar10_resnet18"), "sets": {"id_test": "mini"}}
    arrays = run(cfg, out=str(tmp_path / "out.npz"), device=str(DEVICE))

    encode, tf = legacy_openood(ckpt, "official")
    x = torch.stack([tf(Image.open(tmp_path / "images" / "images_classic" / r).convert("RGB")) for r in rels])
    f_old, lo_old = encode(x)
    np.testing.assert_allclose(arrays["id_test_feats"], f_old.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(arrays["id_test_logits"], lo_old.numpy(), rtol=1e-6, atol=1e-6)
    assert arrays["id_test_labels"].tolist() == list(range(6))


@pytest.mark.parametrize("name", ["cifar10_resnet18", "cifar10_supcon_resnet18", "cifar100_resnet18", "cifar10_dinov2",
                                  "imagenet200_resnet18", "imagenet1k_resnet50", "imagenet1k_resnet50_csid"])
def test_extract_configs_valid(name):
    """Mọi cấu hình trỏ tới backbone đã đăng ký và imglist có thật (nếu có bản OpenOOD của dự án gốc)."""
    cfg = load_config(name)
    assert cfg["backbone"] in available_backbones()
    lists = Path(os.environ.get("CURE_IMGLISTS", OODD / "data" / "benchmark_imglist")) / cfg["imglists"]
    if not lists.exists():
        pytest.skip(f"thiếu imglist {lists}")
    for spec in cfg["sets"].values():
        assert (lists / f"{spec['list'] if isinstance(spec, dict) else spec}.txt").exists()
