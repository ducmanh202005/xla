# M³Diff Studio — cách chạy

Chỉ dùng notebook **M3Diff_Demo_Web_7_O.ipynb**. Bản hiện tại chạy **B128 (LoRA + SCST)**.
Trước lần đầu cần Drive được chia sẻ có model gốc và checkpoint B128 (hoặc adapter đã xuất); xem **CHECKPOINT_VA_CHIA_SE.md**. Ô 2 tự xuất adapter từ checkpoint train khi cần. Không bắt buộc tải trọng số về laptop.
Danh sách notebook huấn luyện/đánh giá: **notebooks/README.md**. Chỉ mở web thì không cần chạy lại các file đó.

## 1. Chuẩn bị Colab

Upload notebook lên Colab, bật GPU rồi chạy **1 → 2 → 3 → 6 → 7**. Chạy từ máy tính khác vẫn được: model chạy trên GPU Colab, còn model gốc/checkpoint nằm trên Drive được chia sẻ. Thành viên cần mount đúng tài khoản và truy cập được `MyDrive/M3Diff_Storage/`. Không cần tải/gửi checkpoint riêng.

- Ô 1: chuẩn bị môi trường.
- Ô 2: cho phép Drive, kiểm tra model gốc, đúng adapter B128 và GPU. Tự xuất adapter từ checkpoint train nếu bạn chưa có gói xuất.
- Ô 3: chọn ảnh trước rồi ảnh sau, chờ **KIỂM TRA ĐẠT**.
- Ô 6: chuẩn bị kết nối web, dùng lại model đã nạp.
- Ô 7: mở kết nối và tải **m3diff_connection.json** về laptop.

Ô 4–5 tùy chọn: so sánh trực tiếp trong Colab và tải ZIP chẩn đoán. Có thể bỏ qua khi dùng web.

### Chưa thấy thư mục demo_scst_B128 trên Drive?

Đây là thư mục xuất riêng cho demo, không phải tên thư mục checkpoint đã huấn luyện. **Bạn không cần tự tạo hoặc huấn luyện lại.** Khi chạy ô 2, notebook kiểm tra checkpoint B128 trong:

`MyDrive/M3Diff_Storage/MDP_Projector_v1/experiments/scst_lora_20261031/`

Nếu đúng checkpoint, ô 2 tự tạo **MyDrive/M3Diff_Storage/demo_scst_B128/** chứa `adapter_model.pt` và `adapter_manifest.json`. Sau khi ô 2 báo **CHECKPOINT B128 ĐẠT**, chạy tiếp ô 3 → 6 → 7. Nếu thiếu checkpoint nguồn, notebook sẽ báo lỗi; tạo thư mục rỗng không thay thế được trọng số.

## 2. Mở web trên laptop

1. Nhấp đúp **CHAY_DEMO.cmd**. Lần đầu cần mạng để cài thư viện giao diện; laptop không tải checkpoint.
2. Mở **http://127.0.0.1:8501**.
3. Mở file **m3diff_connection.json** ở thanh bên trái, bấm **Kết nối / kiểm tra**.
4. Chọn hai ảnh, bấm **So sánh toàn ảnh**. Hoặc khoanh vùng trên cả hai ảnh, xem ảnh cắt rồi bấm **So sánh hai vùng đã chọn**.
5. Bấm **Lưu kết quả** để đưa vào lịch sử; tải HTML/JSON/ZIP trong tab **Lịch sử & báo cáo**.

Giữ Colab và cửa sổ chạy web mở. Mỗi cặp mới chỉ thao tác trên web, không chạy lại ô nạp model. Kiểm tra kết nối không chạy suy luận.

## Khi gặp lỗi

- Colab mất hoàn toàn phiên: chạy lại **1 → 2 → 3 → 6 → 7**, nhập file kết nối mới trên web.
- Chỉ link web hết hoạt động, model còn sống: chạy lại **ô 7**, nhập file kết nối mới.
- Báo chưa có model/runner: chạy **ô 3** rồi **6 → 7** trong cùng notebook.
- Không mở được web: kiểm tra cửa sổ chạy còn mở. Dừng web laptop bằng Ctrl+C trong cửa sổ đó.
- Gửi ảnh lâu: thời gian tổng có cả tải ảnh và chờ mạng. Dùng ảnh vừa đủ rõ và chờ yêu cầu xong trước khi bấm tiếp.

File kết nối chứa mật khẩu: giữ riêng, không đưa vào bài nộp công khai. Muốn đóng riêng kết nối Colab, dùng `web_demo.close()`; muốn giải phóng model, dùng `runner.close()` và cần chạy lại ô 3 trước khi dùng tiếp.

## Phần mới và giới hạn

- Chọn vùng thủ công, xem ảnh cắt, đối chiếu kết quả toàn ảnh/vùng, lưu lịch sử và xuất báo cáo có ảnh. Vị trí trái/phải trong caption vùng là vị trí trong ảnh cắt.
- Lịch sử tại `demo/history/`; HTML tải về mở được khi Colab đã ngắt. Ảnh ví dụ chỉ có nếu thư mục dữ liệu mẫu của dự án vẫn còn.
- Caption tiếng Anh do model gốc kết hợp adapter B128 sinh ra và có thể sai. Kết quả full1570, cách huấn luyện và giới hạn phép so sánh nằm trong **CAI_TIEN_SCST_VA_KET_QUA.md**.
- Mã web/API và notebook được kiểm tra cục bộ. B128 đã có benchmark A100; notebook demo B mới cần xác nhận bằng lượt so sánh ô3 và báo cáo ô5. Không coi benchmark là bằng chứng giao diện đã chạy thành công.

Python laptop cần 3.10 trở lên; môi trường web riêng ở `.runtime/demo-web`. Model dùng Python 3.10 riêng trên Colab. Các file Python còn lại trong thư mục là thành phần hỗ trợ, không cần mở/chạy từng file.
