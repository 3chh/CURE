# BÁO CÁO ĐO LƯỜNG HIỆU NĂNG PHẦN CỨNG THỰC TẾ (CURE OOD)
**Môi trường hệ thống:** Windows 11 | Python 3.12 | PyTorch 2.11 (CUDA 12.8)  
**Phần cứng kiểm thử:** NVIDIA GeForce RTX 3050 6GB Laptop GPU & Intel Core i5/i7 CPU

---

## 1. TỔNG HỢP KẾT QUẢ ĐO ĐẠC THỰC CHỨNG (MICROBENCHMARK)

| Tiêu chí Kỹ thuật | Profile 1: Edge Vision (GPU) | Profile 2: Industrial NIDS (SCADA CPU) |
| :--- | :--- | :--- |
| **Phần cứng thực thi** | NVIDIA GeForce RTX 3050 6GB Laptop GPU | Intel CPU (Single Core, x86_64) |
| **Kích thước Batch** | $B = 128$ (GPU Matmul Batching) | **$B = 1$ (Streaming trực tiếp từng gói tin)** |
| **Số chiều đặc trưng ($d_{\text{feat}}$)** | 512 (ResNet/ViT feature vectors) | 36 (Westermo industrial flow features) |
| **Kích thước Gallery / CDM** | $N_G = 10.000, M = 512$ | $N_G = 2.000, M = 256$ |
| **Độ trễ trung bình (Mean)** | **1.186 ms / batch 128** | **301.7 µs (0.3017 ms) / gói** |
| **Độ trễ P50 (Median)** | 1.105 ms | 210.2 µs |
| **Độ trễ P95 (Đuôi 95%)** | 1.797 ms | 783.3 µs |
| **Độ trễ P99 (Đuôi 99%)** | 2.088 ms | 1.894 ms |
| **Độ trễ phân bổ / mẫu** | **0.0093 ms / mẫu (9.3 µs)** | **0.3017 ms / mẫu** |
| **Mức chiếm dụng VRAM GPU** | **84.3 MB** *(Rất an toàn trên GPU 6GB)* | **0 MB (Thuần CPU)** |
| **Thông lượng suy luận đặc trưng** | **107.888 frames/sec** *(Chỉ bước tính điểm)* | **3.315 packets/sec** *(Python thuần đơn luồng)* |

---

## 2. NHẬN XÉT HỌC THUẬT & GIỚI HẠN THỰC NGHIỆM

1. **Về Profile 1 (Edge Vision GPU):**
   - Độ trễ 1.186 ms cho cả batch 128 mẫu chứng minh bước tính điểm CURE (kNN + CDM + Cauchy) hoàn toàn không phải là điểm nghẽn (bottleneck) của pipeline thị giác.
   - *Lưu ý minh bạch:* Thông lượng 107.888 FPS là tốc độ xử lý vector đặc trưng (feature scoring throughput). Tốc độ toàn hệ thống (end-to-end) sẽ bị chi phối bởi mô hình trích xuất đặc trưng (backbone ResNet/ViT, thường chạy ở 60–120 FPS trên GPU biên).
   - Mức chiếm dụng VRAM chỉ 84.3 MB (nhờ cơ chế chunking động 512) giúp CURE chạy song song mượt mà cùng các mô hình thị giác lớn mà không gây tràn bộ nhớ.

2. **Về Profile 2 (Industrial NIDS SCADA CPU):**
   - Độ trễ 301.7 microsecond (P50: 210 µs) trên một luồng CPU đơn của Python đáp ứng thoải mái các chu kỳ quét (polling cycle) của mạng tự động hóa công nghiệp (Modbus TCP, DNP3, IEC-60870-5-104 thường có nhịp 10–100 ms).
   - *Lưu ý minh bạch:* Thông lượng 3.315 gói/s được đo bằng trình thông dịch Python thuần. Trong các hệ thống mạng chuyển mạch tốc độ cao (Gigabit Ethernet yêu cầu wire-speed $>50.000$ PPS), mô-đun này cần được biên dịch bằng C++/Rust hoặc tích hợp qua eBPF/XDP để tận dụng vector SIMD.
