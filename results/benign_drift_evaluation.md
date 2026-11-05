# Thực Nghiệm Phản Chứng: Đánh Giá Khả Năng Chống Báo Động Giả Khi Trôi Dạt Lành Tính (Benign Covariate Shift)

## 1. Bảng Tổng Hợp Khi 35% Mẫu ID Bị Trôi Dạt / Nhiễu Cảm Biến Môi Trường

| Phương pháp | AUROC (%) ↑ | FPR95 (%) ↓ (Báo động giả) | TPR@FPR5 (%) ↑ | Độ tăng FPR95 so với kNN (càng thấp càng tốt) |
|---|:---:|:---:|:---:|:---:|
| **KNN** | 60.89 | **53.73%** | 1.39% | **+0.00%** |
| **CURE** | 71.98 | **48.29%** | 17.70% | **-5.44%** |
| **CURE_OEBH_B1** | 65.88 | **51.65%** | 9.39% | **-2.08%** |
| **OODD** | 69.71 | **62.05%** | 24.86% | **+8.32%** |
| **WARDEN** | 58.42 | **57.03%** | 0.00% | **+3.30%** |
| **AGE_QUEUE** | 68.90 | **66.33%** | 47.39% | **+12.60%** |
