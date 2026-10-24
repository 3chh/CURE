# Bắt đầu với repo CURE

Hướng dẫn cho thành viên mới: cài đặt, lấy dữ liệu, chạy thử, tái hiện bảng kết quả và thêm phương pháp mới.

## 1. Cài đặt

```bash
git clone https://github.com/3chh/CURE.git && cd CURE
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,extract]"
```

- Python ≥ 3.10, PyTorch ≥ 2.1. Có GPU thì dùng GPU; phương pháp tĩnh bản OpenOOD bắt buộc GPU vì code gốc gọi `.cuda()`.
- Kiểm tra: `cure-ood list` in danh sách phương pháp và benchmark.

## 2. Lấy dữ liệu

Dữ liệu nằm trên Hugging Face, để private vì license ImageNet/Places/TinyImageNet. Ko cần tài khoản HF: xin trưởng nhóm token chỉ đọc (gửi riêng, **ko commit, ko dán vào code hay issue**) rồi đặt biến môi trường một lần:

```bash
setx HF_TOKEN hf_...            # Windows, mở terminal mới sau đó
export HF_TOKEN=hf_...          # Linux/macOS, thêm vào ~/.bashrc
```

Ko cần tải trước: thiếu file đặc trưng, gallery OODD hay checkpoint nào thì lần chạy đầu tự lấy đúng file đó từ HF vào cache HF (`HF_HOME`), lần sau dùng lại. Muốn chạy offline thì đặt `CURE_HF_AUTO=0`. Nếu muốn tải hết về thư mục repo từ đầu:

```bash
python scripts/download_data.py                  # đặc trưng (~2,1 GB) + gallery 5 crop của OODD
python scripts/download_data.py checkpoints      # chỉ cần khi trích lại đặc trưng hoặc train
python scripts/download_data.py imglists images cifar10 cifar100   # ảnh, chỉ khi trích lại đặc trưng
```

| Thư mục | Nội dung | Biến môi trường đổi vị trí |
|---|---|---|
| `data/features/` | file đặc trưng `.npz`, gallery OODD | `CURE_FEATURES` |
| `checkpoints/` | checkpoint backbone | `CURE_CHECKPOINTS` (nhiều thư mục, ngăn bởi `;` trên Windows) |
| `data/images/images_classic/`, `images_largescale/` | ảnh OpenOOD | `CURE_IMAGES` |
| `data/benchmark_imglist/` | imglist OpenOOD | `CURE_IMGLISTS` |

Các thư mục này bị `.gitignore` chặn, ko bao giờ commit dữ liệu hay kết quả.

## 3. Chạy thử

```bash
cure-ood run quick        # 1 luồng, ~1 phút
cure-ood report quick     # results/quick/report.md
pytest                    # test; test cần dữ liệu tự bỏ qua nếu thiếu
```

## 4. Thí nghiệm có sẵn

| Config (`configs/experiments/`) | Nội dung | Thời gian (RTX 3050) |
|---|---|---|
| `main` | Bảng chính vòng 58: 45 nhóm, đối thủ theo code gốc hoặc cài đủ theo bài | ~4 giờ |
| `round57` | Tái hiện bảng chương 7 hiện có trong docx (bản đối thủ cũ) | ~3 giờ |
| `robustness` | Độ bền theo backbone: SupCon, DINOv2 | ~3 giờ |
| `seed35` | Xác nhận trên seed 3-5 | ~4 giờ |
| `ablation` | Bỏ hậu kiểm, oracle, bỏ lối lùi, Fisher, e-LOND, cửa sổ, cỡ CDM | ~1-2 giờ |
| `realtime` | Vòng 59: đánh đổi cỡ batch, độ trễ, chất lượng | ~6 giờ |
| `oodd_in200_alpha1` | Kiểm độ nhạy OODD trên ImageNet-200 | ~10 phút |

Máy 16 GB RAM nên chạy theo nhóm nhỏ để ko hết bộ nhớ:

```bash
python scripts/run_all.py main --group 5      # mỗi (benchmark, tập lạ, 5 phương pháp) một tiến trình, tự thử lại
cure-ood report main
```

Việc nặng nên chạy trên Kaggle thay vì laptop: mở `notebooks/kaggle_round60.ipynb` trên Kaggle, thêm Secrets `HF_TOKEN` và `GH_TOKEN` (token GitHub chỉ đọc), bật Internet, rồi Save & Run All. Notebook tự clone repo, chạy song song và nén kết quả vào tab Output.

Kết quả lưu theo từng luồng và từng phương pháp, chạy lại lệnh sẽ tiếp từ chỗ dừng. Bản tóm tắt kết quả đã chạy nằm trong `reports/`.

## 5. Quy tắc làm việc của nhóm

- **Đăng ký trước:** mọi vòng thử nghiệm mới ghi giả thuyết và ngưỡng đạt vào research log của đồ án **trước khi chạy**, báo cả kết quả trượt. Ko chỉnh tham số sau khi đã thấy kết quả.
- **Phương pháp có code chính thức:** vendor nguyên văn vào `methods/*/_official/` (ghi repo và commit ở đầu file), adapter chỉ đưa dữ liệu vào, mọi chỗ sửa đánh dấu `ADAPTED`, kèm test so với hàm gốc (`tests/test_official_*.py`).
- **Phương pháp ko có code:** cài đủ mọi thành phần của bài, kèm `docs/reimplementations/<tên>.md` đối chiếu từng mục/công thức với code và các điểm bài bỏ ngỏ.
- **Ko làm hỏng số cũ:** `tests/test_golden_*.py` phải qua; nếu thay đổi có chủ đích làm đổi số, giữ bản cũ trong `methods/legacy/` và ghi lý do.
- **Phong cách code:** docstring 1 dòng, comment ngắn giải thích lý do, `ruff check src tests scripts` sạch.

## 6. Thêm phương pháp, backbone, dữ liệu

- Phương pháp mới: [adding_a_method.md](adding_a_method.md).
- Backbone hoặc bộ dữ liệu mới: [adding_a_backbone.md](adding_a_backbone.md).
- Tự huấn luyện backbone: [training.md](training.md).
- Thuật toán CURE: [method.md](method.md).
