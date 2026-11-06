# Báo cáo kiểm định độc lập: CURE, Bộ 1 và Bộ 2

**Phạm vi:** Tôi chỉ kiểm định số liệu trong tin nhắn. Tôi đã thử đối chiếu với file kết quả trong repo nhưng hook `cbm-code-discovery-gate` báo lỗi cú pháp (đường dẫn `Hp Victus` có dấu cách), nên chưa đối chiếu được. Hook này cần sửa trước khi dùng các số này trong báo cáo.

**Kết luận:** Phần số học nhất quán: tôi tính lại trung bình Bộ 2 thì khớp từng chữ số. Nhưng có nhiều điểm hội đồng sẽ hỏi. Hai điểm nặng nhất:
- **Trên Bộ 2, CURE không phải hạng 1.** WARDEN* có AUROC 90.76, FPR95 26.89, Harm 0%, Worst Δ -0.32, đều tốt hơn hoặc bằng CURE.
- **CURE không cải thiện near-OOD ở Bộ 1** (90.95 so với kNN 91.06).

---

## 1. Kiểm định tính chân thực và nhất quán

### Các điểm đã khớp
- Trung bình Bộ 2 khớp. kNN: tổng AUROC 14 kịch bản là 1262.38, chia 14 được 90.17. CURE Online: 90.52. WARDEN*: 90.76. FPR95 của kNN: 28.89.
- Harm Rate quy về số nguyên hợp lệ. Bộ 1 có 126 nhóm: 6, 1, 73, 4, 62, 104 nhóm. Bộ 2 có 14 kịch bản: 2, 12, 5 kịch bản.
- Harm Rate dùng ngưỡng ngầm khoảng 0.5 điểm. Ví dụ CURE Online có Harm 0% nhưng thua kNN ở 6/14 kịch bản (MITM -0.41, PortScan -0.24...). Với AdaODD*, đếm số kịch bản tụt hơn 0.5 thì ra đúng 5/14. Ngưỡng này phải ghi rõ trong định nghĩa. Nếu không, hội đồng sẽ chỉ ra "Harm 0% mà Worst Δ âm" và cho là mâu thuẫn.

### Rủi ro bị bắt bẻ, xếp theo mức nghiêm trọng

| # | Vấn đề | Vì sao nguy hiểm |
|---|---|---|
| 1 | **Lệch phạm vi đã chốt** | `CLAUDE.md` ghi giai đoạn này chỉ làm OpenOOD v1.5, ảnh AI để sau. Westermo, WM-811K, DeepFake đều chưa có đăng ký trước trong `CURE_Causal_Research_Log.md`. Nếu chưa đăng ký thì toàn bộ Bộ 2 chỉ là khám phá, không được trình bày như xác nhận. |
| 2 | **Tên đề tài sai** | "Conformal Unsupervised Test-Time OOD Detection with e-BH Certified Dynamic Memory" khác tên đã chốt: *CURE: Dynamic Memory Curation via Retroactive Eviction for Test-Time OOD Detection*. |
| 3 | **B=1 với e-BH mâu thuẫn kết quả cũ** | Vòng 44 cho thấy e-BH theo đợt với batch 1 kém an toàn (86.7%). Vòng 45 phải chuyển sang e-LOND, và bản đó an toàn nhưng quá thận trọng. Ở đây "Online e-BH, B=1" lại an toàn hơn Batch (Harm 0.8% so với 4.8%). Cần ghi rõ B=1 chạy e-BH hay e-LOND. e-BH trên đúng một mẫu thì bản chất là kiểm định đơn, không còn là "BH". |
| 4 | **WARDEN\* và AdaODD\* đều là bản tái lập** | WARDEN không có mã nguồn. Hơn nữa, vòng 21 từng xác nhận OODD chính thức khá mạnh, còn ở đây OODD thua cả kNN tĩnh (-1.26, Harm 57.9%). Hội đồng sẽ hỏi bản tái lập có đúng siêu tham số của bài gốc không. |
| 5 | **AgeQueue là đối thủ rơm** | AUROC 37.27 trên ResNet-18-Bottleneck là thấp hơn ngẫu nhiên, gần như chắc là lỗi đảo dấu điểm hoặc hàng đợi bị nhiễm hoàn toàn. Worst Δ -80 làm yếu cả bảng. Cũng không được gọi AgeQueue (FIFO) và kNN tĩnh là "SOTA". Thực tế chỉ có 3 phương pháp SOTA (OODD, AdaODD\*, WARDEN\*), không phải 6. |
| 6 | **CURE trùng khít kNN ở nhiều chỗ** | Westermo toàn luồng (Online), IP Spoofing và SynthBuster cho ra 65.44/98.3, 51.23/97.9 và 73.86/73.9, giống kNN đến từng chữ số. Nghĩa là CDM không nạp mẫu nào, CURE không làm gì. "An toàn" ở đây là do kiêng nạp, không phải do tuyển chọn tốt. Phải báo tỉ lệ nạp (admission rate) theo từng kịch bản. |
| 7 | **Trung bình AUROC trộn các miền khác nhau** | Lấy trung bình gói mạng SCADA chung với video Sora-2 không có ý nghĩa thống kê. Nên báo theo miền và dùng xếp hạng trung bình hoặc kiểm định Wilcoxon cặp. |
| 8 | **Thiếu khoảng tin cậy** | Có 3 seed và 126 nhóm nhưng không có độ lệch chuẩn, CI hay p-value. Chênh +0.35 ở Bộ 2 gần chắc chắn không có ý nghĩa thống kê. |
| 9 | **BAD-MISCONF có trong câu hỏi nhưng không có trong bảng** | Không được nêu kết luận về một kịch bản không có số liệu. |
| 10 | **Nguồn dữ liệu DeepFake 2026** | Cần chỉ rõ bộ dữ liệu, cách tách khung hình video, và kiểm tra rò rỉ (cùng prompt hoặc cùng nguồn xuất hiện ở cả ID lẫn OOD). |
| 11 | **WM-811K mâu thuẫn vòng 47-48** | Vòng 47-48 ghi WM-811K trượt D2. Giờ lại +3.2 điểm. Cần giải thích cái gì đã thay đổi. |
| 12 | **Kịch bản "Hard" là nhiễu nhân tạo do mình tự tạo** | σ=0.12, mất 35% kênh, π=2% đều do mình chọn. Phải chứng minh các tham số này chốt trước khi chạy, không được chọn sau khi thấy kết quả. |
| 13 | **Độ trễ và DCR** | Độ trễ trong bảng backbone là của bộ trích đặc trưng, không phải chi phí của CURE. Bộ 2 mới là chi phí thật: Online 319 µs, chậm khoảng 14 lần so với Batch 22.8 µs và khoảng 145 lần so với kNN. DCR cần định nghĩa. Ngoài ra WARDEN\* có DCR 0% nhưng Harm 49.2%, tức tác hại của nó không đến từ nhiễm bẩn bộ nhớ. Điều này cần giải thích. |

---

## 2. Kết luận theo từng metric (viết lại cho trung thực)

| Metric | Bộ 1 (7 backbone) | Bộ 2 (14 kịch bản) | Câu nên viết trong báo cáo |
|---|---|---|---|
| **AUROC** | CURE Batch hạng 1 (93.94, +1.59 so với kNN) | **WARDEN\* hạng 1** (90.76), CURE 90.58/90.52 | "CURE vượt kNN ổn định trên benchmark ảnh; trên Bộ 2, CURE ngang WARDEN\* và kNN." |
| **Near-OOD** | **CURE không thắng** (90.95-91.05 so với kNN 91.06, AdaODD\* 91.08) | - | Khớp vòng 54-56: near-OOD nghẽn ở mật độ cộng với e-BH. Nên nêu thẳng đây là giới hạn. |
| **Far-OOD** | CURE +2.44 (95.43), toàn bộ lợi ích nằm ở đây | - | Lợi ích của CURE là lợi ích far-OOD. |
| **FPR95** | CURE tốt nhất (23.25) | WARDEN\* ≈ CURE Batch (26.89 so với 26.83) | Hai bên tương đương ở Bộ 2. |
| **TPR@FPR5** | CURE +9.6 so với kNN, mức tăng lớn nhất | Ngang nhau (77.7-77.8) | Metric mạnh nhất của CURE ở Bộ 1. |
| **AUPR-Out** | CURE 74.70 tốt nhất | WARDEN\* 82.86 > CURE Online 82.76 | |
| **DCR** | CURE 0.8-3.8% so với OODD 35.5% và AgeQueue 69.1% | CURE Online 0% | Đây là đóng góp thật: cơ chế chứng nhận giữ bộ nhớ sạch. |
| **Harm Rate** | Online 0.8% < AdaODD\* 3.2% < Batch 4.8% ≪ WARDEN\* 49.2% | CURE Online 0% = WARDEN\* 0%, Batch 14.3% | **CURE Batch không "an toàn tuyệt đối".** Nên chọn Online làm cấu hình chính. |
| **Worst Δ** | Online -0.91, tốt nhất trong các phương pháp thích nghi | WARDEN\* -0.32 tốt hơn CURE -0.41 | |
| **Latency/Train** | Backbone quyết định chi phí, CURE gần như miễn phí khi chạy Batch | Online đắt (319 µs) | Cần đánh đổi giữa độ trễ và độ an toàn. |

**Câu chốt an toàn:** *Trong các phương pháp có bộ nhớ động, CURE là phương pháp duy nhất vừa tăng AUROC trung bình vừa giữ Harm ≤ 1% trên cả hai bộ. Trên Bộ 2, WARDEN\* tái lập đạt mức tương đương.* Không nên viết "CURE thắng mọi SOTA".

---

## 3. Đánh đổi giữa các backbone

- **Lợi ích của CURE tỉ lệ nghịch với chất lượng nền.** CURE tăng +2.24 (ResNet-20), +3.11 (MobileNetV2), +1.08-1.89 (ResNet-18), nhưng chỉ +0.59 với DINOv2. Không gian đặc trưng càng tốt thì kNN tĩnh càng gần trần (98.67), bộ nhớ động càng ít việc để làm.
- **Chiều đặc trưng không quyết định.** ResNet-20 64-d (91.91) gần bằng MobileNetV2 1280-d (92.53). Thứ quyết định là chất lượng biểu diễn, không phải số chiều.
- **Tính ổn định giữa các seed tốt.** Ba seed ResNet-18 cho CURE từ 93.24 đến 93.93 và đều thắng kNN. Đây là bằng chứng mạnh nhất của Bộ 1 và nên đưa lên trước.
- **Ngoại lệ cần giải thích:** WARDEN\* thắng CURE trên Bottleneck (93.33 so với 92.83), còn WARDEN\* sụp trên MobileNetV2 (79.54).
- **Khuyến nghị:**
  - *Thiết bị biên, CPU, cần dưới 5 ms:* dùng ResNet-20 cùng CURE. Mô hình 1.1 MB, 3.5 ms trên CPU, đạt 91.9, đây là nơi CURE đem lại nhiều nhất. DINOv2 mất 188 ms trên CPU, không dùng được cho thời gian thực.
  - *Có GPU và cần độ chính xác tối đa:* dùng DINOv2 cùng CURE (99.26). Lý do dùng CURE ở đây là bảo hiểm khi trôi dạt, không phải để tăng AUROC.
  - *Mức giữa:* ResNet-18 512-d, vì cân bằng tốt nhất giữa độ chính xác, độ ổn định giữa các seed và độ trễ GPU khoảng 4 ms.

---

## 4. Vì sao Westermo cơ bản bão hòa còn các kịch bản khó thì phân hóa

**a) Hiệu ứng trần.** Với SSH, PortScan, MITM, tấn công tách biệt rõ trong không gian đặc trưng gói tin, kNN tĩnh đã đạt 99.35-99.95. Không còn chỗ cho bộ nhớ tăng thêm. Mọi phương pháp thích nghi chỉ có thể *giữ nguyên hoặc làm hỏng*. Thực tế **CURE thua kNN nhẹ ở mọi kịch bản Westermo** (FPR95 tăng từ 0-0.3% lên 1.1-2.5%). Phải nói thẳng điều này, không được trình bày như thắng.

**b) Kịch bản khó phân hóa các phương pháp *nạp không kiểm soát*, không phân hóa CURE với kNN.** Với Stealth Boundary, mất 35% kênh và APT π=2%, OODD và AgeQueue sụp (FPR95 35-57% và tới 99.9%). Cơ chế: mẫu OOD gần biên ID, hoặc mẫu nhỏ giọt, bị nạp vào bộ nhớ như ID. Bộ nhớ dần "học" tấn công, gây tự nhiễm bẩn (DCR 39.6% và 80.3%). CURE nhờ kiểm định e-BH có đối chứng nên từ chối nạp, giữ ở mức kNN. **Nhưng WARDEN\* và kNN tĩnh cũng giữ được.** Vì vậy khẳng định đúng là "CURE *tránh được* thảm họa của OODD/AgeQueue", không phải "CURE vượt trội".

**c) Có những thất bại mà không phương pháp nào cứu được.** Westermo toàn luồng (~65) và IP Spoofing (~51, ngang ngẫu nhiên) cho thấy đặc trưng không mã hóa được tín hiệu tấn công. Thiết bị giả mạo có cùng phân phối gói tin. Đây là giới hạn của biểu diễn, không phải của bộ nhớ. Nên đưa vào mục "giới hạn" như một phát hiện trung thực.

**d) Miền thật có trôi dạt là nơi duy nhất CURE thắng rõ.**
- WM-811K Center: 96.51 lên 99.70, FPR95 từ 17.4 xuống 0.2. Đây là trường hợp có lợi rõ nhất.
- WM-811K Scratch: +2.55 đến +4.62, nhưng WARDEN\* (89.10) cao hơn.
- DeepFake: +0.15 đến +0.23, FPR95 giảm 3 lần (0.9 xuống 0.3), trong khi OODD sụp (FPR95 khoảng 15%).
- Cơ chế: ID ở luồng test lệch khỏi gallery huấn luyện (khác đợt wafer, khác generator). Bộ nhớ được chứng nhận nạp đúng những mẫu ID mới này. Đúng khung "khi nào bộ nhớ giúp/hại" đã chốt.
- Tuy vậy, kết quả WM-811K mâu thuẫn vòng 47-48, nên cần chạy xác nhận có đăng ký trước.

---

## Việc cần làm trước buổi báo cáo
1. Sửa hook, đối chiếu từng số với file kết quả gốc. Không đưa số chép tay vào docx.
2. Ghi rõ B=1 dùng e-BH hay e-LOND, ngưỡng Harm Rate, định nghĩa DCR và tỉ lệ nạp mỗi kịch bản.
3. Bổ sung CI qua 3 seed và kiểm định Wilcoxon cặp CURE so với kNN và CURE so với WARDEN\*.
4. Gỡ lỗi AgeQueue (AUROC 37) hoặc bỏ khỏi bảng chính.
5. Đánh dấu Bộ 2 là *khám phá, ngoài phạm vi giai đoạn hiện tại*, trừ khi có đăng ký trước trong research log.
6. Dùng tên đề tài đã chốt.