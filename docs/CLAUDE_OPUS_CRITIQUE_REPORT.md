# Thẩm định kết quả CURE

Tin nhắn của bạn bị lỗi mã hóa và dừng ở dấu hai chấm, không kèm số liệu. Vì vậy tôi đánh giá toàn bộ kết quả hiện có trong repo, gồm các vòng 58-63, `master20_81`, Bộ 1, Bộ 2 và `multiseed_eval` của ngày 06/11.

**Phán quyết: chưa sẵn sàng bảo vệ, cần bổ sung.** Lõi "chứng nhận giữ bộ nhớ sạch, lợi ích ở far-OOD có cụm" là có thật. Tuy nhiên, phần Westermo trong hồ sơ phải rút lại, và cấu hình chính đang không nhất quán. Bảng chính cũng chưa có xác nhận độc lập. Tôi không thấy dấu hiệu bịa số ở bảng chính, nhưng bảng vòng 63 không truy được về file kết quả của nó.

**Giới hạn của lần thẩm định này:** đây là kết quả đọc mã và file, chưa chạy lại gì.
- Python và pytest cần phê duyệt mà phiên này không có, nên cả Westermo lẫn test đều chưa chạy.
- Tôi chưa đọc được `BaoCao_TienDo_ThucNghiem_CURE_OOD.docx` và chưa thẩm định các chứng minh mệnh đề.
- Vì thế, nếu docx đang chép lại các số bị nêu dưới đây thì phải sửa theo.

## Lỗi nghiêm trọng

1. **"OODD sụp còn 35,99%" gần như chắc chắn là lỗi đảo dấu của ta, không phải tự đầu độc.**
   - `scripts/benchmark_network_attacks_cure.py:178` tính `-(s0 - aff)`, trong khi `s0` đã theo hướng OOD. Công thức này đảo ngược điểm kNN.
   - 35,99 xấp xỉ 100 - 65,33. Benchmark khác trên cùng luồng cho OODD 61,64 (`cyber_and_realworld_sota.md`).
   - Vì vậy luận điểm 2 và 5 của vòng 63 không đứng được, kể cả câu "chứng minh Định lý 3".

2. **Bảng vòng 63 mâu thuẫn với chính file JSON nó trích dẫn.**

   | Chỉ số | Bảng trong nhật ký | `westermo_cyber_attack_analysis.json` |
   |---|---|---|
   | Số mẫu SSH | 348 | 524 |
   | e-BH trên SSH | 99,81 | 91,57 |
   | e-BH trên MITM | 99,28 | 78,86 |
   | kNN trên MISCONF | 58,18 | 50,69 |

   - Script không có nhóm `BAD-UNKNOWN`, `BAD-OTHER`, và cũng không xuất OODD theo nhóm.
   - Nhật ký gọi Oracle là "naive", nhưng script chạy Oracle bằng Cauchy.
   - Câu "Online e-BH phát hiện bất ổn định" là sai: `stats.py` không có bộ dò trôi dạt. Nếu CDM rỗng thì chỉ là không đủ bằng chứng, và DCR = 0 khi đó là 0/0 được gán bằng 0.

3. **Cauchy, cấu hình đã chốt, có điểm yếu ở chính vùng Mệnh đề 3 khẳng định.** Đây là suy luận từ mã, cần kiểm chứng bằng chạy thật.
   - Khi `p_mem = 1`, `combine` kẹp về 1 - 1e-12, nên `tan` xấp xỉ -3,2e11. Giá trị này xóa hẳn bằng chứng từ `p_s`, vốn bị chặn dưới bởi 1/(|C|+1).
   - Trường hợp này xảy ra với mẫu lạ không giống gì trong CDM. Rất có thể đây là nguyên nhân của -30,79 trên APT, -14,41 trên MITM và -8,07 trên SSH (`switch_ablation.md`).
   - Test ở `patch_test_stats.py` chỉ lấy `p_mem` trong [0,05; 0,95], nên không chạm tới trường hợp này.

4. **Fisher được đổi sau khi thấy kết quả, và ba bộ đang dùng khác cấu hình chính.**
   - Bộ 1, Bộ 2, trôi dạt lành tính và multiseed đều đặt `rule="fisher"`. `switch_ablation.md` tự ghi "đổi hậu nghiệm".
   - Bảng 81 nhóm vẫn dùng cấu hình đã chốt là Cauchy (trong `methods/` không có "fisher").
   - Trên Bộ 2, đổi sang Fisher chuyển Δ so với kNN từ -3,60 thành +0,42. Như vậy Bộ 2 đã thành dữ liệu phát triển.
   - Nhật ký không có vòng đăng ký nào cho Fisher hay `master20`, tức vi phạm quy tắc trong `CLAUDE.md`.

5. **Chưa có xác nhận độc lập.** "Seed mới" chỉ xáo lại luồng trên cùng ảnh và cùng checkpoint, sau hơn 50 vòng phát triển trên chính các bộ này. 81 nhóm cũng không độc lập, vì chỉ có 27 cặp bộ dữ liệu - tập OOD nhân với 3 mức π. Do đó p < 1e-5 bị thổi phồng; cần bootstrap theo cụm.

## Rủi ro lớn về luận điểm

- **Khoảng cách với OODD bị thổi phồng.** Bảng 81 nhóm tính OODD trên ImageNet-200 là 73,90, trong khi cấu hình ALPHA = 1 đạt 86,00. Thay vào 9 nhóm đó, OODD lên khoảng 91,74, và mức hơn giảm từ 2,79 xuống khoảng 1,45 (ước tính của tôi). Thêm T-out thì còn thấp hơn.
- **Câu chuyện nhân quả "bộ nhớ bẩn gây hại" bị chính ablation phản bác.**
  - Trên Bộ 1, công tắc ngây thơ thắng CURE: 93,57 so với 92,95, DCR 32,9% mà hại 0%.
  - Oracle sạch tuyệt đối vẫn hại 11,1% nhóm ở Bộ 1 và 57,1% ở Bộ 2.
  - Vai trò thực sự của CURE là giảm rủi ro đuôi, không phải tối đa AUROC.
- **"An toàn 100%" phụ thuộc định nghĩa.**
  - Con số này dùng dung sai 0,5, tính trên trung bình 20 seed. Một triển khai thật chỉ thấy một luồng.
  - Theo từng luồng, CURE theo đợt hại tới 5,56%, Δ tệ nhất -1,76 (`multiseed_auroc_summary.json`).
  - Seed 3-5 cho 98,8%; Bộ 2 Cauchy hại 28,6%, Fisher hại 14,3%.
- **Thí nghiệm trôi dạt lành tính không kiểm đúng giả thuyết nguy hiểm.** Nhiễu đẳng hướng từng mẫu không tạo cụm, nên không thể gây chứng nhận nhầm. Kịch bản đáng thử là cụm ID mới tái diễn. Ngoài ra σ cố định làm mức nhiễu khác nhau giữa chiều 64 và 1280.
- **Near-OOD không có lợi ích, và các con số đang lệch nhau:** +0,27, +0,30, -0,11, -0,40, -0,6. Tên "The Cost of Safety" không thuyết phục. Với CDM chỉ khoảng 6 mẫu, đây là bộ nhớ không kích hoạt, không phải cái giá của an toàn.
- **WARDEN không thua trên Bộ 2.** So với CURE, WARDEN có AUROC 90,76 so với 90,58, hại 0%, và nhanh hơn. So sánh CURE - WARDEN có KTC [-0,08; 11,58] chứa 0 nhưng vẫn được gắn "significant"; n ở đây thực chất là 7 backbone, không phải 126 nhóm.

## Lỗi báo cáo cần sửa

- **Bảng near/far toàn cục:** trong `backbone_sota_tradeoff.md`, OODD có near = far = tổng = 91,09, và các đối thủ khác cũng vậy. Điều này vô lý, vì multiseed cho OODD near khoảng 86. Tôi chưa tìm ra nguyên nhân. Câu "AdaODD thắng near" đang dựa trên cột lỗi này.
- **Vòng 61 báo thiếu nhánh:** ResNet-50 và LogitNorm đã đăng ký nhưng không có trong bảng kết quả.
- **DCR:** luồng có CDM rỗng được tính là 0, làm trung bình thấp đi (`eval_multiseed_auroc.py:67`).
- **Pareto d:** "sweet spot d = 256" chỉ dựa trên một luồng SVHN, seed 0. Ablation 45 nhóm lại cho d = 512 cao hơn: 90,13 so với 89,96.
- **Độ trễ:** 107.888 FPS chỉ là bước chấm điểm với gallery 10k; số đo thực ở vòng 59 là 8,1 ms mỗi batch. Ở 30 mẫu/giây, B = 128 có trễ đầu-cuối khoảng 2,1 giây.
- **Rò rỉ tương lai:** test `test_causality_no_lookahead.py` đúng ý tưởng, chỉ cần chạy để xác nhận. Script Westermo lại chạy e-BH trên toàn luồng, tức dùng p-value tương lai và không phải quy trình chính thức. Riêng e-BH và online e-BH trong script này không được coi là đại diện cho CURE.
- **Đối thủ rơm và tên đề tài:** AgeQueue đạt 37,27 trên Bottleneck, không nên tính là SOTA. Các prompt audit cũng dùng sai tên đề tài đã chốt.

## Điều đứng vững

- Hậu kiểm giữ bộ nhớ sạch: DCR 7-11%, so với 75% khi bỏ hậu kiểm.
- Far-OOD tăng rõ, và online e-BH B = 1 an toàn trên luồng đơn: hại 0% ở mọi seed huấn luyện trong multiseed.
- Multiseed tăng +1,84 đến +3,17 so với kNN trên ResNet-20 và MobileNetV2. Đây là mức tăng ổn định nhưng dùng Fisher hậu nghiệm và chỉ trên CIFAR-10.
- Việc báo trung thực các giả thuyết trượt ở vòng 58-62 và kết luận OODD cao hơn CURE trên ImageNet-200 là điểm cộng lớn.

## Việc cần làm trước buổi báo cáo

1. Rút bảng vòng 63 và mọi kết luận Westermo khỏi báo cáo cuối tuần. Sửa dấu ở dòng 178, chuyển sang e-BH theo đợt và sinh lại bảng bằng script.
2. Kiểm chứng giả thuyết `p_mem = 1`. Nếu đúng, đăng ký vòng mới so Cauchy với Fisher (hoặc Cauchy cắt đuôi) trên dữ liệu chưa dùng, rồi mới chốt cấu hình. Sửa cả Mệnh đề 3.
3. Thống nhất một cấu hình cho mọi bảng, tính lại KTC theo cụm 27 cặp, và báo an toàn theo từng luồng.
4. Sửa OODD trên ImageNet-200 và cột near/far, đồng thời bổ sung ResNet-50 và LogitNorm.
5. Chạy thêm kịch bản cụm ID mới tái diễn.

**Luận điểm nên trình bày với Thầy:** "CURE là cơ chế chứng nhận giúp bộ nhớ động giảm rủi ro đuôi và mang lại lợi ích có điều kiện khi OOD tái diễn thành cụm dày." Nên báo thẳng ba giới hạn: near-OOD không có lợi, ImageNet-200 thua OODD, và chưa có xác nhận độc lập.

Nếu bạn duyệt lệnh Python, tôi có thể chạy lại Westermo và test ngay để chốt mục 1 và 3.
