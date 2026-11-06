# Phiếu phản biện độc lập: CURE

**Phán quyết: chưa đủ để bảo vệ xuất sắc, và Reject ở hội nghị top-tier.** Lõi cơ chế có giá trị thật, nhưng hồ sơ đang trình bày nhiều hơn những gì số liệu chứng minh.

*Phạm vi kiểm tra: tôi tự tính lại từ các bảng trong hồ sơ. Chưa đối chiếu được nhật ký nghiên cứu vì công cụ tìm kiếm bị hook `cbm-code-discovery-gate` chặn do lỗi, nên việc Bộ 2 và Fisher có đăng ký trước hay không vẫn **chưa xác minh**.*

## 1. Năm phát hiện từ chính số liệu

| # | Phát hiện | Bằng chứng | Hệ quả |
|---|---|---|---|
| 1 | Lợi thế ở Bộ 2 đến gần như hoàn toàn từ hai kịch bản WM-811K | Hai kịch bản này đóng góp +5.77/+6.13 điểm dương, tức **94%**. Bỏ chúng ra, 12 kịch bản còn lại có Δ trung bình **-0.08** so với kNN | Trên dữ liệu này, CURE không hơn kNN nếu thiếu wafer |
| 2 | Không có lợi ích nào ở SCADA | Trong 9 kịch bản đầu, CURE có 6 lần thua, 3 lần hòa, 0 lần thắng; tổng Δ = **-1.28** | Tiêu đề "tấn công mạng SCADA" gây hiểu nhầm, và 5/14 kịch bản không phải SCADA |
| 3 | Tăng ở Bảng 1 chỉ đến từ far-OOD | Far +4.44, near **-0.11**. AUROC tổng khớp công thức (2 near + 4 far)/6, nên far chiếm trọng số 2/3 | +1.59 phần lớn là hiệu ứng trọng số; near-OOD, phần khó nhất, không được cải thiện |
| 4 | "Harm 0%" có ngưỡng chưa công bố | CURE B=1 thua kNN ở 6/14 kịch bản mà Harm vẫn 0%. Suy ra có dung sai nằm giữa 0.41 và 0.88 | Không công bố ngưỡng thì chỉ số này mất giá trị |
| 5 | AUROC che giấu hại vận hành | FPR95: SSH 0→1.1, MITM 0.3→0.9, Telemetry 0.2→**1.5**, Stealth 1.1→3.0 | Ở SCADA, FPR tăng 3-7 lần khi kNN đã gần tối ưu là hại thật, dù Harm báo 0% |

Hồ sơ còn có lỗi trình bày mà hội đồng sẽ thấy ngay:

- **Tiêu đề Bảng 1 sai:** "7 backbones × 7 SOTA" là 49, không phải 126. Con số này nhiều khả năng là 7 × 6 OOD × 3 tỷ lệ, kèm 3 seed.
- **"7 SOTA" là phóng đại:** bảng gồm 2 biến thể CURE, kNN, FIFO và chỉ 3 đối thủ thật. Trong đó 2 đối thủ mang dấu `*` chưa được giải thích.
- **Bảng 3 và Bảng 1 lệch nhau:** kNN là 91.10 so với 92.35. Các tỷ lệ Harm ở Bộ 1 của Bảng 3 khớp bội số của 1/36, tức ablation chỉ chạy trên tập con khoảng 36 nhóm. Hồ sơ không nói rõ điều này.
- **Thiếu độ bất định:** không có khoảng tin cậy hay kiểm định ghép cặp. Các chênh lệch 0.24-0.42 điểm không thể gọi là khác biệt nếu chưa kiểm định.

## 2. Năm câu hỏi chí mạng

**Câu 1. "Bỏ WM-811K đi, CURE còn hơn kNN không?"**
Câu trả lời trung thực là không: -0.08. Học viên phải nói trước điều này và nêu điều kiện để bộ nhớ có ích: OOD tái diễn, tập trung, và kNN chưa bão hòa. Không nên để hội đồng tự phát hiện.

**Câu 2. "Nếu tự đầu độc gây hại, vì sao công tắc ngây thơ nhiễm 32.9% lại Harm 0% và thắng CURE, còn WARDEN* nhiễm 0% lại Harm 49.2%?"**
Hai dòng này cùng phản bác giả thuyết trung tâm rằng DCR quyết định Harm. Công tắc ngây thơ lấy được 77% khoảng cách tới Oracle, CURE chỉ lấy được 58%. Vậy FDR thấp không đồng nghĩa với AUROC cao.

**Câu 3. "Fisher được chọn trước hay sau khi thấy Cauchy sụp -30.79?"**
Nếu chọn sau, Bộ 2 đã trở thành tập phát triển, và mọi con số Fisher trên đó đều lạc quan. Thêm nữa, 4.01 × 14 ≈ 56 điểm AUROC bị mất, tức chỉ 30.79 đến từ APT; ít nhất **25 điểm** còn lại mất ở kịch bản khác. Câu "sụp ở APT nhỏ giọt" vì vậy chưa giải thích đủ.

**Câu 4. "Viết định lý ra: giả định nào, kết luận gì, bước nào cần hoán đổi?"**
FDR ≤ 0.20 là đảm bảo trung bình về tuyển chọn, không phải đảm bảo về AUROC hay Harm. Giả thuyết không ở đây là gì: "không thuộc cụm" hay "là ID"? Bộ nhớ phụ thuộc quá khứ của luồng, và hồ sơ có điều kiện hóa đúng khi hiệu chuẩn hay không? Fisher cổ điển cần p-value độc lập, trong khi hai p-value ở đây tính trên cùng một mẫu.

**Câu 5. "Trên luồng theo thời gian thật duy nhất, Westermo Chronological, vì sao CURE trùng khít kNN (65.44/98.3)?"**
Rất có thể bộ nhớ không kích hoạt. Khi đó phần lớn tính "an toàn" chỉ là việc không làm gì, và kịch bản Stealth APT cũng vậy. Cần báo cáo tỷ lệ kích hoạt. Không có số đó, hội đồng không thể phân biệt "an toàn có chứng nhận" với "vô hiệu".

## 3. Điểm yếu học thuật

**Near-OOD không tăng.** Đây là giới hạn cấu trúc. Kiểm định dựa trên số đếm vượt mức cần cụm tách khỏi đuôi ID, còn near-OOD chồng lấn với ID. DCR 3.8% so với mức cho phép 20% cho thấy thủ tục bảo thủ hơn cần thiết khoảng 5 lần. Hãy thừa nhận giới hạn này. Đừng biện hộ bằng AUROC tổng.

**Công tắc ngây thơ thắng ở Bộ 1.** CURE chỉ thắng nó ở Bộ 2: Harm 14.3% so với 50%, Worst -0.88 so với -5.27. Luận điểm đúng là đánh đổi: CURE hy sinh hiệu năng trung bình để giảm rủi ro đuôi. Nhưng lập luận đó vững chỉ khi Bộ 2 không đồng thời là nơi chọn Fisher.

**Cauchy và Fisher.** Có một lối thoát hợp lệ: nếu tổ hợp được hiệu chuẩn lại trên C, Fisher chỉ là một **hàm điểm**, và giả định độc lập không còn là điều kiện cho tính hợp lệ. Đổi lại, đây là lựa chọn về năng lực phát hiện, cần được chọn trên dữ liệu phát triển và xác nhận trên dữ liệu độc lập. Giải thích cơ chế cũng phải có chứng minh, không chỉ là suy đoán.

**WARDEN* ở Bộ 2.** WARDEN* không thua CURE ở chỉ số chính nào: AUROC, FPR, TPR, Harm, Worst và độ trễ 12.1 so với 319.2 µs. CURE chỉ hơn ở DCR 0.0 so với 0.6. Tuy vậy, toàn bộ khoảng cách đến từ Scratch: bỏ kịch bản này, CURE nhỉnh hơn 0.03 trên 13 kịch bản, tức là hòa. Điều đáng nghi hơn là cùng một phương pháp có Harm 49.2% ở Bộ 1 nhưng 0% ở Bộ 2. Cần chứng minh bản tái lập trung thành và được tinh chỉnh công bằng.

**Thiết kế Bộ 2 yếu.** 8/14 kịch bản có kNN ≥ 99, tức bão hòa trần. 2 kịch bản gần như ngẫu nhiên (51-65%). Do đó chỉ một phần nhỏ kịch bản thực sự phân biệt được các phương pháp. "Stealth APT" với AUROC 99.91 cũng không hề tàng hình.

**Thiếu kịch bản phản chứng.** Hồ sơ không kiểm tra trôi dạt lành tính, ví dụ cụm ID mới bị chứng nhận thành OOD rồi gây báo động giả. Nó cũng không kiểm tra kẻ tấn công thích nghi. AgeQueue với -31.93 chỉ là đối chứng rơm, không nên dùng làm luận điểm.

## 4. Có cần chạy thêm không?

**Có, nhưng không cần mở rộng benchmark.** Năm việc sau là đủ, theo thứ tự ưu tiên:

1. **Xác nhận độc lập, bắt buộc.** Đăng ký trước, khóa Fisher cùng toàn bộ tham số, rồi chạy trên dữ liệu chưa từng dùng để chọn. Nếu trượt, báo cáo cả kết quả trượt.
2. **Thống kê lại, không cần chạy mới nhiều.** Thêm CI bootstrap theo nhóm, Wilcoxon ghép cặp, công bố ngưỡng Harm, và thêm Harm theo FPR95.
3. **Đo tỷ lệ kích hoạt và kích thước CDM.** Tách kết quả theo "bộ nhớ có hoạt động" và "kịch bản còn dư địa". Đồng thời chuyển ablation sang đủ 126 nhóm.
4. **Kiểm thử phản chứng.** Chạy trôi dạt lành tính, cụm ID mới và đầu độc thích nghi, đồng thời quét d, π, FDR và kiểm chứng FDR thực nghiệm.
5. **Viết định lý đầy đủ và kiểm chứng trung thực WARDEN\*.** Dùng cùng ngân sách tinh chỉnh cho mọi đối thủ. Nêu rõ độ trễ tính trên phần nào của pipeline, vì 2.2 µs gần như chắc chắn chưa gồm trích đặc trưng.

Việc **1, 2 và 3 là điều kiện tối thiểu** trước khi bảo vệ. Việc 4 và 5 quyết định khả năng đạt xuất sắc.

## 5. Điểm số

| Tiêu chí | Điểm | Lý do |
|---|---:|---|
| Tính cấp thiết & đóng góp | **6.0** | Vấn đề có thật, nhưng trục xuất và hàng chờ đã có tiền lệ; phần mới nằm ở chứng nhận thống kê |
| Toán học & phương pháp luận | **5.0** | Khung hiện đại, nhưng khoảng cách từ FDR đến lợi ích chưa được chứng minh |
| Thực nghiệm | **5.5** | Quy mô tốt, Oracle và ablation trung thực; nhưng thiếu CI, có nguy cơ rò rỉ lựa chọn, lợi ích tập trung |
| Ứng dụng thực tế | **4.5** | Không gây hại nặng là thật, nhưng lợi ích trên SCADA bằng 0 và độ trễ B=1 cao gấp 145 lần kNN |
| **Tổng kết** | **5.3** | Chế độ an toàn đáng tin, lợi ích còn hẹp |

| Bối cảnh | Hiện tại | Sau khi làm việc 1-3 | Sau khi làm đủ 1-5 |
|---|---|---|---|
| Đồ án kỹ sư | **8.0 - Đạt tốt** | 8.5-9.0 | 9.0+ |
| Luận văn thạc sĩ | **7.0-7.5 - Weak Accept** | 8.0-8.5 | 8.5-9.0 |
| NeurIPS/ICLR/CVPR | **Reject** | Reject đến Borderline | Borderline, cần định lý mạnh |
| Workshop hoặc hội nghị hạng B | **Borderline** | Weak Accept | Accept |

**Thông điệp nên bảo vệ:** "CURE là cơ chế chứng nhận giúp bộ nhớ động không gây hại nặng, và mang lại lợi ích có điều kiện khi OOD tái diễn thành cụm." Tuyên bố này hẹp nhưng trụ được. Còn tuyên bố "CURE vượt SOTA trên SCADA" thì chính Bảng 2 bác bỏ.

## Ghi chú ngoài vai phản biện

Một số điểm dưới đây lấy từ ghi nhớ dự án, không phải từ hồ sơ, và cũng chưa kiểm tra lại với repo:

- **WM-811K:** ghi nhớ nói nó từng trượt tiêu chí sàng lọc D2 ở vòng 47-48. Nếu đúng, việc 94% lợi ích đến từ đây sẽ là câu hỏi nguy hiểm nhất.
- **Phương pháp chính:** ghi nhớ ghi bản cuối dùng **Cauchy**, còn hồ sơ dùng Fisher. `CLAUDE.md` cũng cấm chỉnh tham số sau khi thấy kết quả xác nhận, nên hãy kiểm tra vòng đăng ký của Fisher.
- **Điều bị bỏ khỏi hồ sơ:** ghi nhớ ghi 10/10 domain thật chưa có vế dương, SPR đã có trục xuất hồi tố từ 2021, MINAS đã có hàng đợi và xóa quá hạn. WARDEN cũng là bản tái lập từ mô tả vì không có mã nguồn. Hội đồng có thể biết các công trình này, nên hãy chủ động trình bày.
- **Lỗi hook:** `cbm-code-discovery-gate` đang hỏng vì đường dẫn có dấu cách (`Hp Victus`) và chặn mọi lệnh `Grep`. Sửa xong, tôi có thể đối chiếu các điểm trên với nhật ký.