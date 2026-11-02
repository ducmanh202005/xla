# M³Diff Studio — checkpoint B128

Demo mô tả thay đổi giữa hai ảnh, chọn vùng, lưu lịch sử và xuất báo cáo. Model chạy trên Colab; Streamlit chạy trên laptop.

- **Notebook liên quan đến cải tiến và vai trò từng file:** [notebooks/README.md](notebooks/README.md).
- **Phương pháp, huấn luyện và kết quả:** [CAI_TIEN_SCST_VA_KET_QUA.md](CAI_TIEN_SCST_VA_KET_QUA.md) — mục 6.1 so với bài báo, mục 6.2 là G/B thực chạy.
- **Chia sẻ notebook và Drive cho nhóm:** [CHECKPOINT_VA_CHIA_SE.md](CHECKPOINT_VA_CHIA_SE.md).
- **Mở web:** [HUONG_DAN.md](HUONG_DAN.md), dùng [M3Diff_Demo_Web_7_O.ipynb](M3Diff_Demo_Web_7_O.ipynb), chạy 1 → 2 → 3 → 6 → 7.
- **Bằng chứng đánh giá:** [results/scst_full1570/](results/scst_full1570/).

Checkpoint B = M³Diff gốc + LoRA bước 128 học bằng CE + SCST. So với Table 3 trong PDF ICCV 2025, số điểm B cao hơn cả bốn chỉ số trên Real và Render. Đây là đối chiếu kết quả công bố; chưa chứng minh riêng SCST gây toàn bộ chênh lệch. Bảng G/B thực chạy vẫn giữ riêng, gồm cả chỉ số giảm và cấu hình sinh câu.

Nhóm chỉ cần chia sẻ notebook và Drive có model gốc/checkpoint đã học; **không cần gửi thêm checkpoint riêng hay tải trọng số về laptop**. Notebook demo tự xuất adapter từ checkpoint B128 khi cần. ZIP báo cáo không chứa trọng số.

Giữ README/LICENSE của [M³Diff gốc](https://github.com/Yuan-Liu-01/M3Diff), được lưu trong source snapshot ở `assets/m3diff_source.zip`.
#### Real — 803 cặp

| Mô hình | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| M³Diff — bài báo | 14,3 | 18,9 | 32,9 | 31,3 |
| B — nhóm | **15,2574** | **19,6437** | **32,9642** | **34,9872** |
| B − bài báo (xấp xỉ) | +0,9574 | +0,7437 | +0,0642 | +3,6872 |

#### Render — 767 cặp

| Mô hình | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| M³Diff — bài báo | 15,7 | 19,9 | 35,3 | 28,3 |
| B — nhóm | **19,4866** | **20,6082** | **35,5738** | **29,3462** |
| B − bài báo (xấp xỉ) | +3,7866 | +0,7082 | +0,2738 | +1,0462 |

"# xla" 
