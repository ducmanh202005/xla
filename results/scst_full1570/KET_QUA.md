# So sánh M³Diff gốc và B — full test1570

Điểm ×100; cùng ảnh, prompt và cách chấm. G dùng repetition_penalty=1,00; B dùng1,05. Chênh lệch gồm tác động của adapter và cấu hình sinh câu. Số dương ở B − G nghĩa là B cao hơn.

| Mô hình | Real BLEU-4 | Real METEOR | Real ROUGE-L | Real CIDEr | Render BLEU-4 | Render METEOR | Render ROUGE-L | Render CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G — penalty 1,00 | 13.3857 | 19.8529 | 32.9185 | 31.7065 | 16.8388 | 20.4169 | 35.5227 | 30.2799 |
| B — penalty 1,05 (LoRA + SCST, bước128) | 15.2574 | 19.6437 | 32.9642 | 34.9872 | 19.4866 | 20.6082 | 35.5738 | 29.3462 |
| B − G | +1.8717 | -0.2092 | +0.0457 | +3.2807 | +2.6477 | +0.1913 | +0.0511 | -0.9336 |

Real: 803 cặp. Render: 767 cặp.

| Toàn bộ 1.570 cặp | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| G — penalty 1,00 | 15.0650 | 20.1319 | 34.1908 | 33.8506 |
| B — penalty 1,05 (LoRA + SCST, bước128) | 17.4916 | 20.1214 | 34.2391 | 35.0356 |
| B − G | +2.4266 | -0.0105 | +0.0483 | +1.1850 |

Real: B tăng 3, giảm 1, bằng 0 chỉ số; thiếu 0.
Render: B tăng 3, giảm 1, bằng 0 chỉ số; thiếu 0.
All: B tăng 3, giảm 1, bằng 0 chỉ số; thiếu 0.

METEOR: complete.
Tập test này đã được sử dụng trước trong dự án; chênh lệch điểm chưa tự chứng minh ý nghĩa thống kê hay hiệu quả trên mọi ảnh. Không huấn luyện hoặc chọn thêm checkpoint trong lượt này.
