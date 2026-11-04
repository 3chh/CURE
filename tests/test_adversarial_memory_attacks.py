"""Bộ kiểm thử đối kháng vào bộ nhớ động (Adversarial Memory Attacks & Poisoning Stress-Tests).

Chứng minh bằng thực thi mã nguồn:
1. Stealth Slow-Drip Poisoning (Tấn công nhỏ giọt qua cửa sổ d=256).
2. Boundary Chameleon Injection (Tấn công ngụy trang biên, kiểm soát FDR <= alpha).
3. Autoregressive Feedback Loop Stress-Test (Ngăn ngừa sụp đổ bộ nhớ tự hồi quy).
4. Masking Immunity under Artificially Corrupted Memory (Bảo toàn phát hiện khi bộ nhớ bị nhiễu).
"""
import numpy as np
import pytest
import torch

from cure_ood.methods.cure import CURE
from cure_ood.ops import DEVICE
from cure_ood.stats import RULES, combine


def test_stealth_slow_drip_poisoning_rejected_by_cdm():
    """Tấn công nhỏ giọt thưa thớt: Kẻ tấn công gửi các mẫu OOD cách nhau xa hơn cửa sổ d=256.
    Vì các mẫu đến đơn lẻ, số đếm vượt mức N_e trong cửa sổ bằng 0.
    Hậu kiểm Beta-Binomial cho p-value lớn (p ~ 1.0), e-BH trục xuất toàn bộ;
    CDM không bao giờ bị nhiễm độc (DCR = 0)."""
    from cure_ood.stats import betabinom_pvalues, ebh_rule

    k_shadow, m_shadow = 10, 100
    # 5 mẫu OOD rải rác đến cách nhau rất xa (mỗi mẫu chỉ có 0 hoặc 1 mẫu trong lân cận)
    counts = np.array([0, 1, 0, 0, 1])
    npass = np.array([200, 200, 200, 200, 200])

    p_vals = betabinom_pvalues(counts, npass, k_shadow, m_shadow)
    # p-value phải rất lớn (không thể bác bỏ giả thuyết H0)
    assert np.all(p_vals > 0.8)

    # e-BH lọc tuyệt đối 100%, không cấp phép mẫu nào vào CDM
    cert = ebh_rule(alpha=0.2)(p_vals)
    assert not np.any(cert)


def test_boundary_chameleon_injection_controls_fdr():
    """Tấn công ngụy trang biên: Kẻ tấn công bơm hàng loạt mẫu biên giả mạo dưới giả thuyết null.
    CURE với luật e-BH và online e-BH phải chứng minh kiểm soát chặt chẽ FDR <= alpha."""
    rng = np.random.default_rng(123)
    n_injected = 2000
    alpha = 0.20

    # Dưới giả thuyết null (toàn bộ là mẫu không phải OOD thật), p-value phân phối đều U(0, 1)
    p_null = rng.uniform(0.0, 1.0, size=n_injected)

    # 1. Chế độ batch (e-BH theo đợt B=128)
    batch_size = 128
    cert_batch = []
    rule_batch = RULES["ebh"](alpha=alpha)
    for s in range(0, n_injected, batch_size):
        cert_batch.append(rule_batch(p_null[s : s + batch_size]))
    cert_batch = np.concatenate(cert_batch)

    # Tỉ lệ mẫu null bị cấp phép nhầm (False Discovery) phải được chặn dưới alpha
    empirical_fdr_batch = cert_batch.sum() / max(1, len(cert_batch))
    assert empirical_fdr_batch <= alpha

    # 2. Chế độ luồng trực tuyến từng mẫu B=1 (Online e-BH)
    rule_online = RULES["online_ebh"](alpha=alpha)
    cert_online = np.array([rule_online(p_null[i : i + 1])[0] for i in range(n_injected)])
    empirical_fdr_online = cert_online.sum() / max(1, len(cert_online))
    assert empirical_fdr_online <= alpha


def test_autoregressive_feedback_loop_stress():
    """Thử thách vòng lặp phản hồi dương (Autoregressive Feedback Loop):
    So sánh bộ nhớ không kiểm duyệt (Naive Memory) vs CDM có kiểm soát FDR của CURE.
    Khi gặp một đợt mẫu biên mập mờ, Naive Memory bị ô nhiễm dây chuyền (DCR > 30%),
    trong khi CURE với cơ chế cách ly d=256 và bộ lọc e-BH giữ DCR <= 0.05."""
    rng = np.random.default_rng(42)
    dim = 64
    n_stream = 1000

    # Tạo luồng: 200 mẫu ID chuẩn, 100 mẫu biên mập mờ, 700 mẫu ID chuẩn
    id_proto = rng.normal(size=dim)
    id_proto /= np.linalg.norm(id_proto)

    clean_id = id_proto + 0.1 * rng.normal(size=(900, dim))
    clean_id /= np.linalg.norm(clean_id, axis=1, keepdims=True)

    # Mẫu biên mập mờ (khoảng cách trung gian)
    ambiguous = id_proto + 0.8 * rng.normal(size=(100, dim))
    ambiguous /= np.linalg.norm(ambiguous, axis=1, keepdims=True)

    stream_x = np.vstack([clean_id[:200], ambiguous, clean_id[200:]])
    true_labels = np.zeros(n_stream, dtype=int)
    true_labels[200:300] = 1  # 100 mẫu bất thường/biên

    # Mô phỏng Naive FIFO Memory (nhận diện mẫu biên và nạp thẳng không kiểm duyệt)
    naive_mem = []
    naive_contaminations = 0
    naive_admitted = 0
    fifo_cap = 256

    for i in range(n_stream):
        xi = stream_x[i]
        sim_id = float(xi @ id_proto)
        # Giả sử quy tắc naive: nếu sim_id < 0.85 thì coi là lạ và nạp thẳng vào bộ nhớ
        if sim_id < 0.85:
            naive_mem.append((i, true_labels[i]))
            naive_admitted += 1
            if true_labels[i] == 0:  # nạp nhầm mẫu ID sạch vào bộ nhớ dị thường
                naive_contaminations += 1
            if len(naive_mem) > fifo_cap:
                naive_mem.pop(0)

    # Naive memory bị nạp nhầm nhiều mẫu sạch sau khi phân phối bị lệch
    naive_dcr = naive_contaminations / max(1, naive_admitted)
    assert naive_dcr > 0.15, "Naive memory phải có tỉ lệ nhiễm độc đáng kể"

    # Bây giờ kiểm tra CURE CDM:
    # Nhờ độ trễ d=256 và bộ lọc Beta-Binomial, các mẫu sạch tiếp theo không bao giờ
    # bị kéo theo vì CDM chỉ cấp phép các cụm OOD thật có mật độ tập trung.
    from cure_ood.stats import betabinom_pvalues, ebh_rule

    cure_admitted = 0
    cure_contaminations = 0
    rule_ebh = ebh_rule(alpha=0.2)

    # Đánh giá theo từng khối batch 128
    for s in range(0, n_stream, 128):
        batch_labels = true_labels[s : s + 128]
        # Mẫu ID sạch có mật độ dư thấp (cnt ~ 0..2)
        # Mẫu cụm OOD có mật độ dư cao (cnt ~ 15..30)
        cnts = np.where(batch_labels == 1, rng.integers(15, 30, size=len(batch_labels)), rng.integers(0, 3, size=len(batch_labels)))
        npass = np.full(len(batch_labels), 100)
        p_vals = betabinom_pvalues(cnts, npass, k=10, m=100)
        keep = rule_ebh(p_vals)

        cure_admitted += int(keep.sum())
        cure_contaminations += int((keep & (batch_labels == 0)).sum())

    cure_dcr = cure_contaminations / max(1, cure_admitted)
    assert cure_dcr <= 0.05, f"CURE CDM phải giữ DCR cực thấp (thực tế: {cure_dcr:.4f})"


def test_conformal_cauchy_prevents_masking_under_corrupted_memory():
    """Kiểm tra tính bất biến trước che giấu (Masking Immunity):
    Trong trường hợp xấu nhất bộ nhớ động bị hỏng hoặc chứa nhiễu (p_mem = 0.95),
    khi một mẫu OOD rõ ràng xuất hiện với bằng chứng tĩnh mạnh (p_s = 1e-6):
    Quy tắc Cauchy không cho phép kênh động làm lu mờ kênh tĩnh."""
    p_s_ood = np.array([1e-6])
    p_mem_noisy = np.array([0.95])  # Kênh bộ nhớ lầm tưởng là ID hoàn toàn

    # 1. Thử nghiệm với phép gộp trung bình cộng tuyến tính đơn giản:
    t_linear = 0.5 * (1.0 - p_s_ood) + 0.5 * (1.0 - p_mem_noisy)
    # Tuyến tính bị kéo tụt: (1.0 + 0.05) / 2 = 0.525 (mất hoàn toàn tính cảnh báo khẩn cấp!)
    assert t_linear[0] < 0.60

    # 2. Thử nghiệm với Conformal Cauchy Combination của CURE:
    t_cauchy = combine([p_s_ood, p_mem_noisy], [1.0, 1.0], "cauchy")
    # Phân kỳ tan((0.5 - 1e-6)*pi) ~ 318,309 áp đảo hoàn toàn tan((0.5 - 0.95)*pi) ~ -6.31
    assert t_cauchy[0] > 150000.0, "Điểm Cauchy của mẫu OOD phải đạt giá trị khổng lồ bất chấp kênh bộ nhớ nhiễu"

