# M³Diff: fine-tune LoRA với SCST và tích hợp demo

## 1. Kết quả đã đạt được

Nhóm có một phương án đã huấn luyện và đánh giá được: **checkpoint B = M³Diff gốc + adapter LoRA học bằng CE kết hợp SCST**, sử dụng cấu hình sinh câu đã chọn. So với **Table 3 của PDF ICCV 2025**, B có số điểm cao hơn cả bốn chỉ số trên Real và Render. Bảng chi tiết, nguồn và giới hạn đối chiếu được trình bày ngay tại **mục 6.1 của tài liệu này**.

Giữ riêng bảng G/B thực chạy để kiểm tra thực nghiệm: trên 1.570 cặp test, cấu hình B tăng 3/4 chỉ số tổng thể so với cấu hình G của nhóm trong lượt đánh giá này. Điểm công bố và điểm G chạy lại là hai nguồn khác nhau, không thay thế lẫn nhau.

Đây là cải thiện đo được **của cấu hình hệ thống đang so sánh**. Chưa thể kết luận toàn bộ mức tăng do riêng SCST: G dùng repetition_penalty 1,00, B dùng 1,05. Thay đổi cấu hình sinh câu cũng ảnh hưởng caption và điểm. Bảng dùng tên ngắn G/B; cấu hình đầy đủ được lưu trong `results/scst_full1570/full_protocol.json`.

Nguồn số liệu: `M3Diff_SCST_Full1570_G100_B105_bao_cao.zip` do nhóm chạy; `summary.json` ghi complete, G và B đều đủ 1.570 cặp. Đã đối chiếu thứ tự ID, ảnh, nhãn tham chiếu giữa hai bản. Số liệu dưới đây lấy từ báo cáo, không phải điểm dự kiến và không phải chấm lại bằng GPU ở máy laptop.

## 2. Kiến trúc và phần nhóm thay đổi

Giữ kiến trúc M³Diff và checkpoint đã học của tác giả. Ảnh trước/sau đi qua bộ mã hóa ảnh SigLIP, MDP xử lý đặc trưng khác biệt, projector đưa đặc trưng sang không gian của mô hình ngôn ngữ Qwen2 trong nền LLaVA-OneVision. Nhóm bổ sung LoRA vào `q_proj` và `v_proj` của các lớp ngôn ngữ.

| Thành phần | Trong lần huấn luyện tạo B128 |
|---|---|
| SigLIP | Đóng băng |
| MDP | Đóng băng, giữ trọng số gốc |
| Projector | Đóng băng, giữ trọng số gốc |
| Trọng số Qwen2 gốc | Đóng băng |
| LoRA ở q_proj/v_proj | Được cập nhật |

Không thiết kế MDP mới, không huấn luyện lại toàn bộ M³Diff từ đầu, không trộn trọng số của các thử nghiệm MDP trước. Adapter B128 là kết quả của thí nghiệm LoRA + SCST riêng. Trong quá trình nạp, cờ `unfreeze_mm_vision_tower=True` phục vụ đường khởi tạo đã dùng trong evaluator; sau đó mã chủ động đóng băng trọng số gốc. Không diễn giải cờ nạp này thành việc fine-tune SigLIP.

LoRA biểu diễn phần điều chỉnh trọng số bằng hai ma trận nhỏ; trọng số gốc được giữ lại. Nhờ đó chỉ cần lưu và chia sẻ adapter riêng. Phương pháp nền: [LoRA — Hu và cộng sự](https://arxiv.org/abs/2106.09685).

## 3. SCST hoạt động như thế nào?

Với mỗi cặp ảnh huấn luyện:

1. Tính loss cross-entropy (CE) với caption tham chiếu, để mô hình tiếp tục học cách mô tả đúng theo nhãn.
2. Sinh một caption bằng lấy mẫu và một caption bằng greedy từ chính mô hình hiện tại.
3. Chấm cả hai bằng reward CIDEr-D so với caption tham chiếu. Dùng cùng bộ thống kê tần suất từ/cụm từ cố định, xây dựng từ tập train.
4. Tính advantage: `reward(caption lấy mẫu) − reward(caption greedy)`.
5. Dùng advantage để tăng hoặc giảm xác suất của chuỗi token đã lấy mẫu, kết hợp với CE rồi cập nhật LoRA.

Loss trong mã: `L = L_CE + 0,05 × L_SCST`, với `L_SCST = −advantage × tổng log xác suất token của caption lấy mẫu`. Reward được tách khỏi gradient. Không huấn luyện thêm một mạng critic riêng; caption greedy đóng vai trò mốc so sánh. Đây là áp dụng kỹ thuật SCST có sẵn vào bài toán mô tả khác biệt, không phải đề xuất thuật toán SCST mới. Tham khảo: [Self-Critical Sequence Training for Image Captioning, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.pdf).

Reward huấn luyện là CIDEr-D ở thang raw của bộ reward. Không nhầm reward này với con số CIDEr nhân 100 trong bảng đánh giá cuối. Mục tiêu reward thiên về mô tả gần nhãn tham chiếu; không bảo đảm tất cả chỉ số hay mọi loại ảnh đều tăng.

## 4. Huấn luyện cụ thể

| Thiết lập | Giá trị đã ghi trong protocol/log |
|---|---|
| Dữ liệu | OmniDiff, nhãn của tác giả; không bổ sung dữ liệu OCR hoặc nhãn mới trong lần này |
| Cặp dùng để cập nhật | 512 cặp train: 256 Real + 256 Render |
| Development | 200 cặp: 100 Real + 100 Render |
| Confirmation | 200 cặp: 100 Real + 100 Render |
| Tập tạo thống kê reward | 12.452 cặp train chính thức: 7.007 Real + 5.445 Render |
| Số cập nhật optimizer | 128; mỗi cập nhật tích lũy gradient từ 4 cặp |
| Số lượt qua 512 cặp đã chọn | Một lượt trong pilot; không phải một epoch toàn bộ 12.452 cặp |
| Optimizer / learning rate | AdamW / 0,00001 |
| Weight decay / gradient clipping | 0 / norm tối đa 1,0 |
| LoRA rank / alpha / dropout | 16 / 32 / 0 |
| Seed | 20261031 |
| Độ dài sinh tối đa | 256 token |
| Thiết bị đã chạy | NVIDIA A100-SXM4-80GB |
| Số học | BF16 autocast; MDP, projector và LoRA FP32 |
| Lưu checkpoint | Mỗi 5 cập nhật, khi kết thúc và khi dừng mềm đủ thời gian lưu |

Nhánh **A** học bằng CE; nhánh **B** học bằng CE + SCST. Hai nhánh bắt đầu từ cùng trạng thái adapter khởi tạo của thí nghiệm. **B không phải lấy adapter A đã học rồi học tiếp**. Các cập nhật smoke được bỏ trước pilot. Checkpoint được lựa chọn cho demo là **B ở bước 128**, không phải nhánh A hoặc checkpoint bước 125.

`train_B.json` trong ZIP huấn luyện ghi tổng thời gian các bước cập nhật B là **3.858,28 giây ≈ 64,30 phút**; peak allocated memory ghi nhận tại thời điểm báo cáo khoảng **20,15 GiB**. Đây không phải thời gian trọn notebook: chưa gồm toàn bộ cài đặt, nạp model, smoke, đánh giá, gián đoạn và nhánh A. A ghi 317,29 giây cho các bước cập nhật.

Các log và checkpoint gồm optimizer/RNG/thứ tự dữ liệu để tiếp tục huấn luyện. Gói chia sẻ demo đã loại các phần này; gói demo chỉ phục vụ suy luận. Muốn tiếp tục train phải giữ thư mục thí nghiệm đầy đủ trên Drive.

## 5. Công nghệ sử dụng

| Phần | Công nghệ |
|---|---|
| Model nền/mã nguồn | M³Diff; snapshot mã đã dùng trong thí nghiệm, có SHA256 cố định |
| Huấn luyện/suy luận | Python 3.10, PyTorch 2.2.2 + CUDA 12.1, Transformers 4.40.0 |
| Adapter | PEFT 0.10.0, LoRA q_proj/v_proj |
| Tiền xử lý ảnh | Pillow, xử lý ảnh pad theo M³Diff, SigLIP đúng revision |
| Tokenizer | Tokenizers 0.19.1; bố cục Qwen giống evaluator |
| Đánh giá | pycocoevalcap 1.2, PTB tokenizer, METEOR chạy Java; BLEU-4 / METEOR / ROUGE-L / CIDEr |
| Môi trường | Google Colab GPU; uv tạo venv Python 3.10 riêng với kernel Colab |
| Giao diện | Streamlit trên laptop; Gradio API trên Colab; một worker giữ model trong GPU |

Mã tác giả: [Yuan-Liu-01/M3Diff](https://github.com/Yuan-Liu-01/M3Diff). Snapshot đi kèm trong `assets/m3diff_source.zip` giữ README và LICENSE gốc. Các phần LoRA/SCST, kiểm tra artifact, báo cáo và demo của nhóm được đặt ngoài thư mục clone.

## 6. Kết quả full test

### 6.1. So sánh B với kết quả công bố trong bài báo

Nguồn: **Table 3, trang 7 PDF (trang kỷ yếu 21446)** của [OmniDiff — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_OmniDiff_A_Comprehensive_Benchmark_for_Fine-grained_Image_Difference_Captioning_ICCV_2025_paper.pdf). **M³Diff — bài báo** là điểm tác giả công bố; **B — nhóm** là checkpoint LoRA + SCST bước 128, lấy từ báo cáo đánh giá full 1.570 cặp đã chạy.

Các chỉ số cùng thang ×100. Chênh lệch = **B − điểm công bố**, đơn vị điểm. Điểm trong PDF có một chữ số thập phân nên chênh lệch dưới đây là xấp xỉ so với giá trị đã làm tròn.

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

B có số điểm cao hơn các giá trị công bố ở cả bốn chỉ số trên Real và Render. **Bài báo không công bố điểm All**, vì vậy không tạo điểm All của tác giả bằng cách lấy trung bình Real/Render. Điểm All của nhóm được trình bày ở mục 6.2.

Đây là đối chiếu với kết quả công bố giữa các lượt đánh giá. Chưa xác nhận mọi chi tiết môi trường, số học, tiền xử lý và cấu hình sinh câu trùng hoàn toàn với lượt chạy của tác giả; điểm G chạy lại của nhóm cũng khác PDF. Vì vậy bảng này chưa cô lập được tác động riêng của SCST. Bảng G/B thực chạy bên dưới được giữ để truy xuất kết quả, gồm cả các chỉ số giảm.

Số liệu B lấy từ [summary.json của lượt full1570](results/scst_full1570/summary.json); điểm tác giả lấy từ Table 3 của PDF được dẫn ở đầu mục này.

### 6.2. So sánh G và B do nhóm thực chạy

Các chỉ số trình bày theo thang nhân 100 của evaluator. Delta là **B − G**, đơn vị điểm, không phải phần trăm tương đối. CIDEr có thể lớn hơn 100. Điểm All được tính trên toàn tập, không lấy trung bình hai điểm CIDEr Real/Render.

#### All — 1.570 cặp

| Bản | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| G | 15,0650 | **20,1319** | 34,1908 | 33,8506 |
| B | **17,4916** | 20,1214 | **34,2391** | **35,0356** |
| B − G | +2,4266 | −0,0105 | +0,0483 | +1,1850 |

#### Real — 803 cặp

| Bản | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| G | 13,3857 | **19,8529** | 32,9185 | 31,7065 |
| B | **15,2574** | 19,6437 | **32,9642** | **34,9872** |
| B − G | +1,8717 | −0,2092 | +0,0457 | +3,2807 |

#### Render — 767 cặp

| Bản | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| G | 16,8388 | 20,4169 | 35,5227 | **30,2799** |
| B | **19,4866** | **20,6082** | **35,5738** | 29,3462 |
| B − G | +2,6477 | +0,1913 | +0,0511 | −0,9336 |

Trong lượt này, 526 caption G đúng cấu hình được dùng lại từ lần trước, 1.044 G được sinh tiếp; B sinh đủ 1.570 caption ở cấu hình mới. Đã kiểm tra cache/protocol để không trộn caption khác cấu hình. 1.570 là số **cặp ảnh**, không phải số file ảnh độc lập; các cặp dùng lại ảnh và toàn test cần 319 ảnh duy nhất.

## 7. Kết luận có thể trình bày

> Nhóm fine-tune M³Diff bằng LoRA, kết hợp cross-entropy và SCST với reward CIDEr-D, rồi tích hợp checkpoint B128 vào ứng dụng so sánh ảnh. Đối chiếu Table 3 của PDF ICCV 2025, B đạt CIDEr Real 34,9872 so với 31,3 công bố và CIDEr Render 29,3462 so với 28,3 công bố; BLEU-4, METEOR, ROUGE-L cũng cao hơn các giá trị công bố trên cả hai tập con. Đây là đối chiếu kết quả công bố giữa các lượt đánh giá, chưa cô lập riêng tác động SCST.

**Trong bảng G/B thực chạy**, không kết luận “tăng mọi chỉ số”: CIDEr Render giảm 0,9336; METEOR Real và All giảm. Khi nói cao hơn cả bốn chỉ số Real/Render phải ghi rõ mốc so sánh là **số công bố trong PDF**. Chưa có kiểm định ý nghĩa thống kê hoặc nhiều seed. Test đã được xem và sử dụng qua nhiều lần thử, không còn là một đánh giá độc lập chưa từng được quan sát. Chưa đủ cơ sở tuyên bố vượt phương pháp gốc trong mọi điều kiện hoặc tăng chắc chắn trên ảnh ngoài tập dữ liệu.

## 8. Đưa vào demo và lưu trữ

Demo mới nạp **model gốc + adapter B128**, dùng đúng đường token/ảnh/số học/sinh câu của B trong full1570. Không ghép lại LoRA vào model gốc và không sửa bốn shard gốc. Nạp một lần trong mỗi worker; những cặp mới dùng lại model. Crop ảnh là chức năng giao diện bổ sung, không phải kiến trúc định vị thay đổi mới.

Các notebook trực tiếp liên quan đã đặt tại [notebooks/](notebooks/README.md). Đọc `CHECKPOINT_VA_CHIA_SE.md` để chia sẻ notebook và Drive cho nhóm; không cần tải/gửi thêm checkpoint riêng. Trọng số adapter thực tế vẫn nằm trên Drive của người chạy; ZIP báo cáo không chứa chúng. Notebook đã được cập nhật và kiểm tra cục bộ; lượt chạy demo B trên A100 cần xác nhận bằng ô 3 và ZIP ô 5, không được coi kết quả benchmark là bằng chứng web đã chạy thành công.
