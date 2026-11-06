# NGHIÊN CỨU ĐÁNH ĐỔI THỰC NGHIỆM: ĐỘ DÀI CỬA SỔ CÁCH LY $d$ (CURE OOD)
**Tập thực nghiệm:** CIFAR-10 vs SVHN (π = 0.2, seed = 0)  
**Mục tiêu:** Khảo sát độ nhạy của tham số độ trễ hàng chờ $d$ tới khả năng tích lũy bằng chứng thống kê và hiệu năng OOD.

---

## 1. BẢNG KẾT QUẢ THỰC NGHIỆM ĐA ĐIỂM $d$

| Độ trễ Hàng chờ $d$ | AUROC (%) | FPR95 (%) | Số mẫu Chứng nhận ($N_{\text{cert}}$) | Bộ nhớ Đệm RAM | Thời gian Chạy | Đánh giá Trạng thái |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$d = 32$** | 95.75% | 16.69% | 4 | 64.0 KB | 4.84 s *(warmup)* | *Quá ngắn:* Thiếu số đếm, bộ lọc e-BH dè dặt |
| **$d = 64$** | 96.70% | 13.59% | 58 | 128.0 KB | 0.75 s | Bắt đầu nhận diện cụm OOD |
| **$d = 128$** | 97.08% | 13.72% | 221 | 256.0 KB | 0.73 s | Tiệm cận tối ưu |
| **$d = 256$ (Mặc định)** | **97.19%** | **13.36%** | **542** | **512.0 KB** | **2.56 s** | **Điểm bão hòa tối ưu (Plateau)** |
| **$d = 512$** | 96.96% | 14.67% | 1034 | 1024.0 KB | 1.10 s | Bão hòa; độ trễ hàng chờ tăng gấp đôi |

---

## 2. PHÂN TÍCH BẢN CHẤT HỌC THUẬT

1. **Vùng $d < 128$ (Thiếu hụt số đếm thống kê):**
   - Phân phối Beta-Binomial cần một cửa sổ quan sát đủ rộng để tích lũy số đếm vượt mức $N_e \ge k_{\text{shadow}} = 10$.
   - Khi $d=32$, chỉ có 4 mẫu được cấp phép vào CDM vì các mẫu OOD đến sau chưa kịp tích lũy trong cửa sổ ngắn, khiến AUROC bị hụt 1.44%.
2. **Vùng $d = 256$ (Điểm bão hòa tối ưu):**
   - Với $d=256$, số mẫu được cấp phép đạt 542 mẫu, AUROC đạt đỉnh 97.19% và FPR95 đạt 13.36%.
   - Chi phí lưu đệm hàng chờ trong RAM chỉ chiếm 512 KB, hoàn toàn không gây áp lực tài nguyên.
3. **Vùng $d \ge 512$ (Hiện tượng bão hòa):**
   - Tăng lên $d=512$ làm số mẫu được chứng nhận tăng lên 1.034 mẫu nhưng AUROC không tăng thêm (96.96%), trong khi độ trễ đưa mẫu vào bộ nhớ tăng gấp đôi.
   - Do đó, $d=256$ là điểm chọn hợp lý cân bằng giữa tính đầy đủ của bằng chứng thống kê và độ trễ hàng chờ.
