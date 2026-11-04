"""Đánh giá toàn diện CURE so với các phương pháp SOTA (kNN, OODD, AdaODD*, WARDEN*, AgeQueue)
trên 15 Kịch bản Tấn công Mạng Công nghiệp (Westermo SCADA/Modbus), Bán dẫn (WM-811K),
Cảm biến Nhiễu (CIFAR-10-C) và Giám định AI Sinh ảnh Hiện đại (OpenFake / SynthBuster).

Bao gồm:
- Kịch bản A: Từng loại tấn công mạng cô lập & hỗn hợp trên Westermo (SSH, PortScan1, PortScan2, MITM, DupIP, Full Mixed)
- Kịch bản B: Tấn công mạng Độ khó cao / Ẩn mình trong Trôi dạt Cấu hình / Nhiễu điện từ / Mất gói SCADA (Giải đáp câu hỏi: Khi bài toán khó hơn, các SOTA ứng xử thế nào?)
- Kịch bản C: Mở rộng đa miền thực tế (Bán dẫn WM-811K, AI Forensics OpenFake 2026 & SynthBuster, Camera mờ/sương mù CIFAR-10-C)

Chạy:
    python deliverables/CURE_repo/scripts/benchmark_cyber_and_realworld_sota.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cure_ood.data.gallery import ExactIndex
from cure_ood.eval.metrics import dcr_counts, detection_metrics
from cure_ood.methods.base import build
from cure_ood.ops import DEVICE, l2


def embed(X: np.ndarray, fit: np.ndarray, mode: str = "lift") -> np.ndarray:
    """lift: làm trắng + nâng lên mặt cầu (dữ liệu bảng); l2: chuẩn hóa L2."""
    if mode == "l2":
        return l2(X)
    mu, sd = X[fit].mean(0), X[fit].std(0)
    k = sd > 1e-9
    Z = (X[:, k] - mu[k]) / sd[k]
    ev, U = np.linalg.eigh(np.cov(Z[fit].T))
    ok = ev > 1e-6 * ev.max()
    Zw = Z @ U[:, ok] / np.sqrt(ev[ok])
    s = 10 * np.quantile(np.linalg.norm(Zw[fit], axis=1), 0.99)
    Y = np.c_[Zw / s, np.ones(len(Zw))]
    return (Y / np.linalg.norm(Y, axis=1, keepdims=True)).astype(np.float32)


@dataclass
class CustomBench:
    name: str
    train_feats: np.ndarray
    cal_feats: np.ndarray
    k: int = 10
    has_logits: bool = False
    split: dict | None = field(default_factory=lambda: {"custom": True})
    t_out: str | None = None

    def __post_init__(self):
        self.G = l2(self.train_feats)
        self.C = l2(self.cal_feats)
        self.G_device = torch.as_tensor(self.G, device=DEVICE)
        self.gallery_index = ExactIndex(self.G, g=self.G_device)
        self.k_knn = int(self.k)
        self.dim = int(self.G.shape[1])

    def knn(self, q: np.ndarray, k: int | None = None, chunk: int = 2048) -> np.ndarray:
        return self.gallery_index.kth_sim(q, k or self.k_knn, chunk)


@dataclass
class CustomStream:
    bench: CustomBench
    domain: str
    feats: np.ndarray
    y: np.ndarray
    near: bool = False
    cache: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = l2(self.feats)
        self.y = np.asarray(self.y, dtype=np.uint8)
        self.logits = None

    def __len__(self) -> int:
        return len(self.y)


METHODS = {
    "knn": {"method": "knn"},
    "cure": {"method": "cure", "rule": "fisher"},
    "cure_oebh_b1": {"method": "cure", "certifier": "online_ebh", "batch": 1, "rule": "fisher"},
    "oodd": {"method": "oodd", "gallery": "approx"},
    "adaodd": {"method": "adaodd"},
    "warden": {"method": "warden", "k": 10},
    "age_queue": {"method": "age_queue", "ttl": 512},
}

METHOD_LABELS = {
    "knn": "kNN (Static)",
    "cure": "CURE (Batch e-BH, B=128)",
    "cure_oebh_b1": "CURE (Online e-BH, B=1)",
    "oodd": "OODD (CVPR)",
    "adaodd": "AdaODD* (ICLR)",
    "warden": "WARDEN* (ICML)",
    "age_queue": "AgeQueue (FIFO)",
}


def make_stream_mix(id_pool: np.ndarray, ood_pool: np.ndarray, pi: float, seed: int, bursty: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Tạo luồng kiểm định với tỉ lệ bất thường pi (ngẫu nhiên i.i.d hoặc theo cụm burst thực tế)."""
    rng = np.random.default_rng(seed)
    n_id_avail, n_ood_avail = len(id_pool), len(ood_pool)
    need_ood = int(round(pi / (1.0 - pi) * n_id_avail))
    if need_ood <= n_ood_avail:
        n_id = n_id_avail
        n_ood = need_ood
    else:
        n_ood = n_ood_avail
        n_id = min(n_id_avail, max(500, int(round(n_ood * (1.0 - pi) / pi))))

    id_idx = rng.choice(n_id_avail, n_id, replace=False)
    ood_idx = rng.choice(n_ood_avail, n_ood, replace=False)
    f_id, f_ood = id_pool[id_idx], ood_pool[ood_idx]

    if not bursty:
        feats = np.r_[f_id, f_ood]
        y = np.r_[np.zeros(n_id, dtype=np.uint8), np.ones(n_ood, dtype=np.uint8)]
        perm = rng.permutation(len(y))
        return feats[perm], y[perm]
    else:
        # Chèn OOD thành các đợt tấn công (bursts cỡ 32-128 gói) xen kẽ trong luồng ID
        feats_list, y_list = [], []
        id_ptr, ood_ptr = 0, 0
        while id_ptr < n_id or ood_ptr < n_ood:
            step_id = min(n_id - id_ptr, int(rng.integers(120, 360)))
            if step_id > 0:
                feats_list.append(f_id[id_ptr:id_ptr + step_id])
                y_list.append(np.zeros(step_id, dtype=np.uint8))
                id_ptr += step_id
            if ood_ptr < n_ood and (id_ptr >= n_id or rng.random() < 0.45):
                step_ood = min(n_ood - ood_ptr, int(rng.integers(32, 128)))
                feats_list.append(f_ood[ood_ptr:ood_ptr + step_ood])
                y_list.append(np.ones(step_ood, dtype=np.uint8))
                ood_ptr += step_ood
        return np.concatenate(feats_list, axis=0), np.concatenate(y_list, axis=0)


def build_scenarios() -> list[dict]:
    scenarios = []

    # 1. Load Westermo Industrial Network Dataset
    w_path = ROOT / "data" / "processed" / "westermo" / "westermo_bottom.npz"
    w_data = np.load(w_path, allow_pickle=True)
    X_w = w_data["X"].astype(np.float64)
    y_w = w_data["y"]
    ref_w = w_data["ref"].astype(bool)
    types_w = w_data["types"]

    # 1A. Chia theo trình tự thời gian gốc (Chronological Split) cho kịch bản Toàn luồng thời gian thực
    ref_idx_chrono = np.flatnonzero(ref_w & (y_w == 0))
    n_fit_chrono = int(0.8 * len(ref_idx_chrono))
    fit_idx_c, cal_idx_c = ref_idx_chrono[:n_fit_chrono], ref_idx_chrono[n_fit_chrono:]
    stream_idx_c = np.flatnonzero(~ref_w)

    Z_w_chrono = embed(X_w, fit_idx_c, "lift")
    gal_w_c, cal_w_c = Z_w_chrono[fit_idx_c], Z_w_chrono[cal_idx_c]
    st_x_c = Z_w_chrono[stream_idx_c]
    st_y_c = (y_w[stream_idx_c] > 0).astype(np.uint8)

    # Kịch bản 1: Toàn luồng thời gian thực Westermo (28,286 gói, có trôi dạt khái niệm BAD-MISCONF)
    scenarios.append({
        "id": "westermo_full_chrono",
        "category": "A. Mạng Công Nghiệp (Gốc & Từng Đòn Tấn Công)",
        "label": "Westermo Toàn Luồng Thời Gian Thực (28.286 gói)",
        "bench": CustomBench("cifar10_westermo_full", gal_w_c, cal_w_c, k=10),
        "streams": [(st_x_c, st_y_c)],
    })

    # 1B. Chia khả hoán (Exchangeable Normal Split) trên nền gói tin bình thường (y_w == 0) để đánh giá từng đòn tấn công & nhiễu
    all_normal_idx = np.flatnonzero(y_w == 0)
    rng_w = np.random.default_rng(2026)
    perm_norm = rng_w.permutation(len(all_normal_idx))
    fit_idx = all_normal_idx[perm_norm[:6000]]
    cal_idx = all_normal_idx[perm_norm[6000:8000]]
    test_norm_idx = all_normal_idx[perm_norm[8000:18000]]

    Z_w = embed(X_w, fit_idx, "lift")
    gal_w, cal_w = Z_w[fit_idx], Z_w[cal_idx]
    normal_pool = Z_w[test_norm_idx]

    ssh_pool = Z_w[types_w == "BAD-SSH"]
    ps1_pool = Z_w[types_w == "BAD-PORTSCAN1"]
    ps2_pool = Z_w[types_w == "BAD-PORTSCAN2"]
    ps_all_pool = Z_w[np.isin(types_w, ["BAD-PORTSCAN1", "BAD-PORTSCAN2"])]
    mitm_pool = Z_w[types_w == "BAD-MITM"]
    misconf_pool = Z_w[types_w == "BAD-MISCONF"]
    dupip_pool = Z_w[types_w == "BAD-MISCONF-DUPLICATION"]
    true_attacks_pool = Z_w[np.isin(types_w, ["BAD-SSH", "BAD-PORTSCAN1", "BAD-PORTSCAN2", "BAD-MITM"])]

    # Kịch bản 2-5: Từng loại tấn công mạng với nhiều seed
    for s_id, s_label, pool, pi in [
        ("westermo_ssh", "Westermo — SSH Brute Force (BAD-SSH)", ssh_pool, 0.12),
        ("westermo_portscan", "Westermo — Port Scanning (BAD-PORTSCAN 1&2)", ps_all_pool, 0.10),
        ("westermo_mitm", "Westermo — Man-in-the-Middle (BAD-MITM)", mitm_pool, 0.10),
        ("westermo_dupip", "Westermo — IP Spoofing / Duplicate Device", dupip_pool, 0.15),
    ]:
        st_list = [make_stream_mix(normal_pool[:6000], pool, pi=pi, seed=sd, bursty=False) for sd in range(3)]
        scenarios.append({
            "id": s_id,
            "category": "A. Mạng Công Nghiệp (Gốc & Từng Đòn Tấn Công)",
            "label": s_label,
            "bench": CustomBench(f"cifar10_{s_id}", gal_w, cal_w, k=10),
            "streams": st_list,
        })

    # Kịch bản B (Độ khó cao):
    # B1: Tấn công mạng dưới nhiễu cảm biến / suy hao kênh truyền (Feature Attenuation + Cluster Shift)
    attenuated_streams = []
    for sd in range(3):
        rng = np.random.default_rng(10 + sd)
        fx, fy = make_stream_mix(normal_pool[:5000], true_attacks_pool, pi=0.18, seed=sd, bursty=False)
        # Suy hao kênh truyền làm các gói tấn công bị kéo gần về phía biên ID nhưng vẫn giữ cấu trúc cụm nội tại
        id_centroid = gal_w[:200].mean(axis=0, keepdims=True)
        fx_att = np.where(fy[:, None] == 1, 0.52 * fx + 0.48 * id_centroid, fx)
        attenuated_streams.append((l2(fx_att), fy))
    scenarios.append({
        "id": "westermo_attenuated_stealth",
        "category": "B. Mạng Công Nghiệp Độ Khó Cao (Ngụy Trang, Nhiễu, Mất Gói & APT)",
        "label": "Westermo Hard — Tấn Công Ngụy Trang Biên ID (Stealth Boundary)",
        "bench": CustomBench("cifar10_w_stealth_bound", gal_w, cal_w, k=10),
        "streams": attenuated_streams,
    })

    # B2: Tấn công mạng dưới nhiễu điện từ công nghiệp (Gaussian Jitter trên cả Gallery/Cal/Stream)
    rng_n = np.random.default_rng(100)
    gal_w_n = l2(gal_w + rng_n.normal(0.0, 0.12, size=gal_w.shape).astype(np.float32))
    cal_w_n = l2(cal_w + rng_n.normal(0.0, 0.12, size=cal_w.shape).astype(np.float32))
    noisy_streams = []
    for sd in range(3):
        rng = np.random.default_rng(110 + sd)
        fx, fy = make_stream_mix(normal_pool[:5000], true_attacks_pool, pi=0.18, seed=sd, bursty=False)
        fx_n = l2(fx + rng.normal(0.0, 0.12, size=fx.shape).astype(np.float32))
        noisy_streams.append((fx_n, fy))
    scenarios.append({
        "id": "westermo_noisy_scada",
        "category": "B. Mạng Công Nghiệp Độ Khó Cao (Ngụy Trang, Nhiễu, Mất Gói & APT)",
        "label": "Westermo Hard — Nhiễu Điện Từ SCADA (SNR Thấp σ=0.12)",
        "bench": CustomBench("cifar10_w_noisy", gal_w_n, cal_w_n, k=10),
        "streams": noisy_streams,
    })

    # B3: Mất gói / Khuyết thuộc tính mạng (35% feature channels bị rơi đồng nhất trên cảm biến thu thập)
    rng_l = np.random.default_rng(200)
    sensor_mask = (rng_l.random((1, gal_w.shape[1])) > 0.35).astype(np.float32)
    gal_w_l = l2(gal_w * sensor_mask + 1e-6)
    cal_w_l = l2(cal_w * sensor_mask + 1e-6)
    lossy_streams = []
    for sd in range(3):
        fx, fy = make_stream_mix(normal_pool[:5000], true_attacks_pool, pi=0.18, seed=sd, bursty=False)
        fx_lossy = l2(fx * sensor_mask + 1e-6)
        lossy_streams.append((fx_lossy, fy))
    scenarios.append({
        "id": "westermo_packet_loss",
        "category": "B. Mạng Công Nghiệp Độ Khó Cao (Ngụy Trang, Nhiễu, Mất Gói & APT)",
        "label": "Westermo Hard — Mất 35% Kênh Cảm Biến Telemetry",
        "bench": CustomBench("cifar10_w_lossy", gal_w_l, cal_w_l, k=10),
        "streams": lossy_streams,
    })

    # B4: Tấn công APT Nhỏ giọt Tốc độ Thấp (Stealth Low-Rate APT pi=0.02)
    scenarios.append({
        "id": "westermo_stealth_apt",
        "category": "B. Mạng Công Nghiệp Độ Khó Cao (Ngụy Trang, Nhiễu, Mất Gói & APT)",
        "label": "Westermo Hard — Tấn Công APT Nhỏ Giọt (Stealth π = 2%)",
        "bench": CustomBench("cifar10_w_apt", gal_w, cal_w, k=10),
        "streams": [make_stream_mix(normal_pool[:6000], true_attacks_pool[:200], pi=0.02, seed=sd, bursty=False) for sd in range(3)],
    })

    # Kịch bản C: Đa miền thực tế khác (Bán dẫn WM-811K, AI Forensics OpenFake 2026, SynthBuster, CIFAR-10-C)
    # C1: Bán dẫn WM-811K
    wm_path = ROOT / "data" / "openood_features" / "wm811k_25x27_dinov2.npz"
    if wm_path.exists():
        wm = np.load(wm_path, allow_pickle=True)
        X_wm = l2(wm["X"].astype(np.float32))
        lbl_wm = wm["label"]
        id_wm = X_wm[lbl_wm == "none"]
        ood_wm_center = X_wm[lbl_wm == "Center"]
        ood_wm_edge = X_wm[np.isin(lbl_wm, ["Edge-Loc", "Loc", "Scratch", "Edge-Ring", "Near-full"])]
        rng_wm = np.random.default_rng(42)
        perm_id = rng_wm.permutation(len(id_wm))
        gal_wm, cal_wm, test_id_wm = id_wm[perm_id[:8000]], id_wm[perm_id[8000:10000]], id_wm[perm_id[10000:]]

        scenarios.append({
            "id": "wm811k_center_defect",
            "category": "C. Đa Miền Thực Tế (Bán Dẫn, AI Forensics & Cảm Biến Nhiễu)",
            "label": "Bán Dẫn WM-811K — Lỗi Tâm Wafer (Center Defect)",
            "bench": CustomBench("cifar10_wm_center", gal_wm, cal_wm, k=10),
            "streams": [make_stream_mix(test_id_wm, ood_wm_center, pi=0.20, seed=sd, bursty=False) for sd in range(3)],
        })
        scenarios.append({
            "id": "wm811k_mixed_defect",
            "category": "C. Đa Miền Thực Tế (Bán Dẫn, AI Forensics & Cảm Biến Nhiễu)",
            "label": "Bán Dẫn WM-811K — Lỗi Xước & Vành (Scratch/Edge-Loc)",
            "bench": CustomBench("cifar10_wm_edge", gal_wm, cal_wm, k=10),
            "streams": [make_stream_mix(test_id_wm, ood_wm_edge, pi=0.12, seed=sd, bursty=False) for sd in range(3)],
        })

    # C2: OpenFake Modern AI Generators (Flux.2, Midjourney-7, Sora-2, Veo-3, GPT-Image-2 vs ImageNet Real)
    of_path = ROOT / "data" / "openood_features" / "openfake_core_modern.npz"
    if of_path.exists():
        of = np.load(of_path, allow_pickle=True)
        X_of = l2(of["clip_L12"].astype(np.float32))
        mod_of = of["model"]
        id_of = X_of[mod_of == "imagenet"]
        ood_of_mj_flux = X_of[np.isin(mod_of, ["midjourney-7", "flux.2-klein-9b", "gpt-image-1.5", "gpt-image-2"])]
        ood_of_video = X_of[np.isin(mod_of, ["sora-2", "veo-3", "wan-video-2.5", "seedream-v5.0"])]
        rng_of = np.random.default_rng(77)
        p_of = rng_of.permutation(len(id_of))
        gal_of, cal_of, test_id_of = id_of[p_of[:6000]], id_of[p_of[6000:7500]], id_of[p_of[7500:]]

        scenarios.append({
            "id": "openfake_mj7_flux2",
            "category": "C. Đa Miền Thực Tế (Bán Dẫn, AI Forensics & Cảm Biến Nhiễu)",
            "label": "DeepFake Forensics — Midjourney-7 / Flux.2 / GPT-Image",
            "bench": CustomBench("cifar10_of_img", gal_of, cal_of, k=10),
            "streams": [make_stream_mix(test_id_of, ood_of_mj_flux, pi=0.25, seed=sd, bursty=False) for sd in range(3)],
        })
        scenarios.append({
            "id": "openfake_sora2_veo3",
            "category": "C. Đa Miền Thực Tế (Bán Dẫn, AI Forensics & Cảm Biến Nhiễu)",
            "label": "DeepFake Video Gen — Sora-2 / Veo-3 / Wan-2.5",
            "bench": CustomBench("cifar10_of_vid", gal_of, cal_of, k=10),
            "streams": [make_stream_mix(test_id_of, ood_of_video, pi=0.25, seed=sd, bursty=False) for sd in range(3)],
        })

    # C3: SynthBuster Forensics (DALLE-2, GLIDE, Stable Diffusion vs RAISE Camera Raw)
    sb_path = ROOT / "data" / "openood_features" / "synthbuster_raise.npz"
    if sb_path.exists():
        sb = np.load(sb_path, allow_pickle=True)
        real_sb = l2(sb["dino_real_L12"].astype(np.float32))
        fake_sb = l2(np.r_[sb["dino_dalle2_L12"], sb["dino_glide_L12"], sb["dino_stable-diffusion-1-4_L12"]].astype(np.float32))
        rng_sb = np.random.default_rng(99)
        p_sb = rng_sb.permutation(len(real_sb))
        gal_sb, cal_sb, test_id_sb = real_sb[p_sb[:300]], real_sb[p_sb[300:420]], real_sb[p_sb[420:]]
        scenarios.append({
            "id": "synthbuster_diffusion",
            "category": "C. Đa Miền Thực Tế (Bán Dẫn, AI Forensics & Cảm Biến Nhiễu)",
            "label": "SynthBuster — Stable Diffusion / DALLE-2 vs Camera Raw",
            "bench": CustomBench("cifar10_sb", gal_sb, cal_sb, k=5),
            "streams": [make_stream_mix(test_id_sb, fake_sb[:400], pi=0.35, seed=sd, bursty=False) for sd in range(3)],
        })

    return scenarios


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    scenarios = build_scenarios()
    print(f"Đã khởi tạo {len(scenarios)} kịch bản thực nghiệm đa miền (Mạng công nghiệp, Độ khó cao, Bán dẫn, DeepFake)!", flush=True)

    results_out = {"scenarios": {}, "summary_by_category": {}, "global_summary": {}}
    cat_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    global_agg = defaultdict(lambda: defaultdict(list))

    for sc in scenarios:
        sc_id = sc["id"]
        sc_cat = sc["category"]
        sc_label = sc["label"]
        bench = sc["bench"]
        streams = sc["streams"]

        print(f"\n------------------------------------------------------------")
        print(f"[{sc_cat}] {sc_label} ({len(streams)} streams)")
        print(f"------------------------------------------------------------", flush=True)

        arms = {}
        for mkey, mcfg in METHODS.items():
            cfg = dict(mcfg)
            mname = cfg.pop("method")
            if mname == "warden":
                cfg["k"] = bench.k_knn
            arms[mkey] = build(mname, **cfg).fit(bench)

        sc_metrics = defaultdict(lambda: defaultdict(list))
        for s_idx, (fx, fy) in enumerate(streams):
            stream_obj = CustomStream(bench=bench, domain=sc_id, feats=fx, y=fy, near=False)
            for mkey, arm in arms.items():
                t0 = time.perf_counter()
                out = arm.score(stream_obj)
                us_per_pkt = (time.perf_counter() - t0) / len(stream_obj) * 1e6
                det = detection_metrics(stream_obj.y, out.scores)
                sc_metrics[mkey]["auroc"].append(100.0 * det["auroc"])
                sc_metrics[mkey]["fpr95"].append(100.0 * det["fpr95"])
                sc_metrics[mkey]["tpr5"].append(100.0 * det["tpr_fpr5"])
                sc_metrics[mkey]["aupr"].append(100.0 * det["aupr_out"])
                sc_metrics[mkey]["us"].append(us_per_pkt)

                if out.memory is not None:
                    dcr_c = dcr_counts(out.memory, stream_obj.y)
                    dcr_pct = 100.0 * dcr_c[0] / dcr_c[1] if len(dcr_c) >= 2 and dcr_c[1] > 0 else 0.0
                    sc_metrics[mkey]["dcr"].append(dcr_pct)

        knn_auroc = float(np.mean(sc_metrics["knn"]["auroc"]))
        sc_summary = {}
        for mkey in METHODS:
            m_auroc = float(np.mean(sc_metrics[mkey]["auroc"]))
            m_fpr95 = float(np.mean(sc_metrics[mkey]["fpr95"]))
            m_tpr5 = float(np.mean(sc_metrics[mkey]["tpr5"]))
            m_aupr = float(np.mean(sc_metrics[mkey]["aupr"]))
            m_us = float(np.mean(sc_metrics[mkey]["us"]))
            m_dcr = float(np.mean(sc_metrics[mkey]["dcr"])) if sc_metrics[mkey]["dcr"] else None
            delta = m_auroc - knn_auroc

            sc_summary[mkey] = {
                "auroc": round(m_auroc, 2),
                "fpr95": round(m_fpr95, 2),
                "tpr_fpr5": round(m_tpr5, 2),
                "aupr_out": round(m_aupr, 2),
                "dcr": round(m_dcr, 2) if m_dcr is not None else None,
                "delta_knn": round(delta, 2),
                "us_per_pkt": round(m_us, 1),
            }
            print(f"  {METHOD_LABELS[mkey]:28s}: AUROC={m_auroc:6.2f}% (Δ={delta:+6.2f}%) | FPR95={m_fpr95:6.2f}% | TPR@5%={m_tpr5:6.2f}% | DCR={f'{m_dcr:.1f}%' if m_dcr is not None else '-':>6s}", flush=True)

            for k_metric, val in [("auroc", m_auroc), ("fpr95", m_fpr95), ("tpr5", m_tpr5), ("aupr", m_aupr), ("delta", delta), ("us", m_us)]:
                cat_agg[sc_cat][mkey][k_metric].append(val)
                global_agg[mkey][k_metric].append(val)
            if m_dcr is not None:
                cat_agg[sc_cat][mkey]["dcr"].append(m_dcr)
                global_agg[mkey]["dcr"].append(m_dcr)

        results_out["scenarios"][sc_id] = {
            "category": sc_cat,
            "label": sc_label,
            "n_streams": len(streams),
            "stream_len": len(streams[0][1]),
            "methods": sc_summary,
        }

    # Aggregate by category and globally
    for cat, mdict in cat_agg.items():
        results_out["summary_by_category"][cat] = {}
        for mkey, vals in mdict.items():
            deltas = vals["delta"]
            results_out["summary_by_category"][cat][mkey] = {
                "auroc": round(float(np.mean(vals["auroc"])), 2),
                "fpr95": round(float(np.mean(vals["fpr95"])), 2),
                "tpr_fpr5": round(float(np.mean(vals["tpr5"])), 2),
                "aupr_out": round(float(np.mean(vals["aupr"])), 2),
                "dcr": round(float(np.mean(vals["dcr"])), 2) if vals["dcr"] else None,
                "delta_knn": round(float(np.mean(deltas)), 2),
                "harm_rate": round(float(np.mean([d < -0.5 for d in deltas]) * 100.0), 1),
                "worst_delta": round(float(np.min(deltas)), 2),
            }

    for mkey, vals in global_agg.items():
        deltas = vals["delta"]
        results_out["global_summary"][mkey] = {
            "label": METHOD_LABELS[mkey],
            "auroc": round(float(np.mean(vals["auroc"])), 2),
            "fpr95": round(float(np.mean(vals["fpr95"])), 2),
            "tpr_fpr5": round(float(np.mean(vals["tpr5"])), 2),
            "aupr_out": round(float(np.mean(vals["aupr"])), 2),
            "dcr": round(float(np.mean(vals["dcr"])), 2) if vals["dcr"] else None,
            "delta_knn": round(float(np.mean(deltas)), 2),
            "harm_rate": round(float(np.mean([d < -0.5 for d in deltas]) * 100.0), 1),
            "worst_delta": round(float(np.min(deltas)), 2),
            "us_per_pkt": round(float(np.mean(vals["us"])), 1),
        }

    out_json = REPO_ROOT / "results" / "cyber_and_realworld_sota.json"
    out_json.write_text(json.dumps(results_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build Markdown report
    md = [
        "# Thực Nghiệm Mở Rộng: So Sánh CURE Với Các SOTA Trên 14 Kịch Bản Tấn Công Mạng Công Nghiệp, Kịch Bản Độ Khó Cao & Đa Miền Thực Tế",
        "",
        "## 1. Bảng Tổng Hợp Toàn Cục Trên Tất Cả 14 Kịch Bản Thực Tế (Mạng SCADA, Nhiễu/Trôi Dạt, Bán Dẫn WM-811K, DeepFake 2026)",
        "",
        "| Phương pháp | AUROC (%) ↑ | FPR95 (%) ↓ | TPR@FPR5 (%) ↑ | AUPR-Out (%) ↑ | DCR (%) ↓ | Δ kNN (%) | Harm Rate (%) ↓ | Worst Δ (%) | Tốc độ (µs/mẫu) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for mkey in ("knn", "cure", "cure_oebh_b1", "oodd", "adaodd", "warden", "age_queue"):
        g = results_out["global_summary"][mkey]
        dcr_s = f"{g['dcr']:.1f}%" if g["dcr"] is not None else "-"
        md.append(
            f"| **{g['label']}** | {g['auroc']:.2f} | {g['fpr95']:.2f} | {g['tpr_fpr5']:.2f} | "
            f"{g['aupr_out']:.2f} | {dcr_s} | {g['delta_knn']:+.2f} | **{g['harm_rate']:.1f}%** | "
            f"{g['worst_delta']:+.2f} | {g['us_per_pkt']:.1f} µs |"
        )

    md.extend([
        "",
        "## 2. Phân Rã Chi Tiết Theo Từng Kịch Bản (AUROC % / FPR95 % / DCR %)",
        "",
        "| Kịch bản thực nghiệm | kNN Tĩnh | **CURE (Batch)** | **CURE (Online B=1)** | OODD (CVPR) | AdaODD* (ICLR) | WARDEN* (ICML) | AgeQueue (FIFO) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for sc in scenarios:
        sc_id = sc["id"]
        r = results_out["scenarios"][sc_id]
        m = r["methods"]
        md.append(
            f"| **{r['label']}** | {m['knn']['auroc']:.2f} / {m['knn']['fpr95']:.1f} | "
            f"**{m['cure']['auroc']:.2f} / {m['cure']['fpr95']:.1f}** | "
            f"**{m['cure_oebh_b1']['auroc']:.2f} / {m['cure_oebh_b1']['fpr95']:.1f}** | "
            f"{m['oodd']['auroc']:.2f} / {m['oodd']['fpr95']:.1f} | "
            f"{m['adaodd']['auroc']:.2f} / {m['adaodd']['fpr95']:.1f} | "
            f"{m['warden']['auroc']:.2f} / {m['warden']['fpr95']:.1f} | "
            f"{m['age_queue']['auroc']:.2f} / {m['age_queue']['fpr95']:.1f} |"
        )

    out_md = REPO_ROOT / "results" / "cyber_and_realworld_sota.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[DONE] Đã lưu báo cáo vào {out_md} và {out_json}", flush=True)


if __name__ == "__main__":
    main()
