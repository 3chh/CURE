# -*- coding: utf-8 -*-
"""
plot_academic_figures.py
-------------------------
Tạo 4 biểu đồ khoa học chuẩn IEEE/ACM (300 DPI, Tiếng Anh toàn bộ, Font Serif/Sans thanh lịch):
1. fig1_training_dynamics.png: Training Loss & Validation Accuracy qua 100 Epochs (ResNet-20, ResNet-18, ResNet-50, MobileNetV2).
2. fig2_tradeoff_latency_vs_auroc.png: Scatter plot Đánh đổi giữa Độ trễ CPU (ms) và AUROC (%), kích thước bóng = dung lượng checkpoint (MB).
3. fig3_cure_gain_vs_capacity.png: Biểu đồ cột thể hiện Quy luật Bù trừ Cận biên Giảm dần (Gain của CURE giảm dần khi năng lực mạng tăng).
4. fig4_multi_seed_variance.png: Error-bar plot thể hiện Định luật Giảm Phương Sai (kNN vs CURE qua 3 seeds độc lập).
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'figure.autolayout': True,
    'figure.dpi': 300
})

OUT_DIR = r"c:\Users\Hp Victus\Desktop\DeepFakeDetection\deliverables\CURE_repo\deliverables\figures"
os.makedirs(OUT_DIR, exist_ok=True)
TRAIN_DIR = r"c:\Users\Hp Victus\Desktop\DeepFakeDetection\deliverables\CURE_repo\results\train"

def load_train_log(model_name):
    log_path = os.path.join(TRAIN_DIR, model_name, "log.jsonl")
    epochs, losses, val_accs = [], [], []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    epochs.append(d.get("epoch", 0))
                    losses.append(d.get("loss", 0))
                    val_accs.append(d.get("val_acc", 0) * 100 if d.get("val_acc", 0) <= 1.0 else d.get("val_acc", 0))
                except Exception:
                    pass
    return epochs, losses, val_accs

# =====================================================================
# FIGURE 1: TRAINING DYNAMICS (Loss & Validation Accuracy)
# =====================================================================
def plot_figure_1():
    models = [
        ("r61_resnet20_s0", "ResNet-20 (64-d)", "#2E86AB"),
        ("r61_resnet18_s0", "ResNet-18 (512-d)", "#2BAE66"),
        ("r61_resnet50", "ResNet-50 (2048-d)", "#E74C3C"),
        ("r61_mobilenetv2", "MobileNetV2 (1280-d)", "#F39C12")
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    for m_id, label, color in models:
        ep, loss, vacc = load_train_log(m_id)
        if ep:
            ax1.plot(ep, loss, label=label, color=color, linewidth=1.8, alpha=0.9)
            ax2.plot(ep, vacc, label=label, color=color, linewidth=1.8, alpha=0.9)
            
    ax1.set_title("(a) Cross-Entropy Training Loss", fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch (1 - 100)")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    
    ax2.set_title("(b) Validation Top-1 Accuracy (%)", fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch (1 - 100)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(40, 100)
    ax2.axhline(94.0, color="#E74C3C", linestyle=":", alpha=0.5, label="ResNet-50 Best (94.0%)")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9)
    
    out_path = os.path.join(OUT_DIR, "fig1_training_dynamics.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")

# =====================================================================
# FIGURE 2: TRADE-OFF PARETO FRONTIER (CPU Latency vs AUROC)
# =====================================================================
def plot_figure_2():
    data = [
        # (Name, CPU Latency ms, AUROC %, Ckpt Size MB, Color, Annotation Offset)
        ("ResNet-20 + CURE (B=1)", 3.50, 91.90, 1.1, "#2E86AB", (10, -10)),
        ("ResNet-20 (Static kNN)", 3.50, 90.20, 1.1, "#A5D8FF", (10, -15)),
        ("MobileNetV2 + CURE (B=1)", 11.32, 91.75, 9.0, "#F39C12", (10, 5)),
        ("ResNet-18 + CURE (B=1)", 10.60, 93.47, 42.7, "#2BAE66", (10, -5)),
        ("ResNet-18 (Official OpenOOD)", 10.60, 92.40, 42.7, "#95A5A6", (-140, 10)),
        ("ResNet-50 + CURE (B=1)", 22.28, 93.74, 90.0, "#E74C3C", (-140, -15)),
        ("ResNet-50 (Static kNN)", 22.28, 92.50, 90.0, "#F1948A", (-130, 8)),
    ]
    
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    
    for name, lat, auroc, size, color, (ox, oy) in data:
        scatter_size = np.sqrt(size) * 45 + 50
        ax.scatter(lat, auroc, s=scatter_size, color=color, alpha=0.85, edgecolors='black', linewidth=1.2)
        ax.annotate(
            f"{name}\n({auroc:.2f}%, {lat:.1f}ms, {size:.1f}MB)",
            xy=(lat, auroc), xytext=(ox, oy), textcoords="offset points",
            fontsize=8.5, fontweight="bold" if "CURE" in name else "normal",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1, alpha=0.85)
        )
        
    # Draw arrow showing Industrial Sweet Spot
    ax.annotate(
        "Industrial Sweet Spot:\n3.5ms CPU, 1.1MB, 91.9% AUROC\n(39x smaller, 3x faster than RN18)",
        xy=(3.50, 91.90), xytext=(5.5, 93.2),
        arrowprops=dict(facecolor="#2E86AB", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9, fontweight="bold", color="#1B365D",
        bbox=dict(boxstyle="round,pad=0.4", fc="#EBF5FB", ec="#2E86AB", lw=1.2)
    )

    ax.set_title("Inference Latency vs OOD Detection Performance (162 OpenOOD Streams)", fontweight="bold", pad=12)
    ax.set_xlabel("CPU Inference Latency per Sample (Batch = 1) [milliseconds]")
    ax.set_ylabel("Global OOD Detection AUROC (%)")
    ax.set_xlim(1.5, 27)
    ax.set_ylim(89.5, 94.5)
    
    out_path = os.path.join(OUT_DIR, "fig2_tradeoff_latency_vs_auroc.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")

# =====================================================================
# FIGURE 3: INVERSE MARGINAL UTILITY OF DYNAMIC MEMORY
# =====================================================================
def plot_figure_3():
    archs = ["MobileNetV2\n(1280-d)", "ResNet-20\n(64-d)", "ResNet-18\n(512-d)", "ResNet-50\n(2048-d)"]
    static_scores = [89.82, 90.20, 92.19, 92.50]
    cure_scores   = [91.75, 91.90, 93.47, 93.74]
    gains         = [1.93, 1.70, 1.28, 1.24]
    
    x = np.arange(len(archs))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={'width_ratios': [2, 1.2]})
    
    # Left: Grouped Bars
    rects1 = ax1.bar(x - width/2, static_scores, width, label="Static kNN Baseline", color="#BDC3C7", edgecolor="black")
    rects2 = ax1.bar(x + width/2, cure_scores, width, label="CURE Online B=1 (e-BH)", color="#2E86AB", edgecolor="black")
    
    ax1.set_ylabel("AUROC (%)")
    ax1.set_title("(a) Performance Comparison across Backbones", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(archs)
    ax1.set_ylim(87, 95)
    ax1.legend(loc="upper left", frameon=True)
    
    # Right: Gain Curve (Diminishing Returns)
    ax2.plot(archs, gains, marker="o", color="#E74C3C", linewidth=2.2, markersize=8)
    for i, g in enumerate(gains):
        ax2.annotate(f"+{g:.2f}%", (i, g), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#C0392B")
    ax2.set_ylabel("Absolute AUROC Gain (Δ %)")
    ax2.set_title("(b) Inverse Marginal Utility", fontweight="bold", pad=10)
    ax2.set_ylim(0.8, 2.3)
    
    out_path = os.path.join(OUT_DIR, "fig3_cure_gain_vs_capacity.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")

# =====================================================================
# FIGURE 4: MULTI-SEED VARIANCE REDUCTION (kNN vs CURE)
# =====================================================================
def plot_figure_4():
    models = ["ResNet-20 (64-d)", "MobileNetV2 (1280-d)", "ResNet-18 (512-d)"]
    
    knn_means = [90.20, 89.82, 92.19]
    knn_stds  = [0.61, 0.37, 0.61]
    
    cure_means = [91.90, 91.75, 93.47]
    cure_stds  = [0.34, 0.13, 0.35]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    
    ax.bar(x - width/2, knn_means, width, yerr=knn_stds, capsize=6, label="Static kNN Baseline", color="#E0E6ED", edgecolor="#7F8C8D", linewidth=1.2)
    ax.bar(x + width/2, cure_means, width, yerr=cure_stds, capsize=6, label="CURE Online B=1 (e-BH Certified)", color="#1B365D", edgecolor="black", linewidth=1.2)
    
    # Annotate variance reduction
    reductions = ["-44% Var", "-65% Var", "-43% Var"]
    for i in range(len(models)):
        ax.annotate(
            f"σ: {knn_stds[i]:.2f}% → {cure_stds[i]:.2f}%\n({reductions[i]})",
            xy=(i, cure_means[i] + cure_stds[i] + 0.35),
            ha="center", fontsize=8.5, fontweight="bold", color="#1B365D"
        )
        
    ax.set_title("Variance Reduction Theorem: Standard Deviation Across 3 Independent Seeds", fontweight="bold", pad=12)
    ax.set_ylabel("AUROC (%) across 3 Seeds (Mean ± Std)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(88, 95.5)
    ax.legend(loc="upper left", frameon=True)
    
    out_path = os.path.join(OUT_DIR, "fig4_multi_seed_variance.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")

def main():
    print("Generating Academic Figures (300 DPI, English)...")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    print("All 4 academic figures generated successfully!")

if __name__ == "__main__":
    main()
