# Hai notebook chính của phương án LoRA + SCST

Thư mục này chỉ giữ hai notebook chính: **huấn luyện LoRA + SCST** và **đánh giá full 1.570 cặp**. Đây là bản sao nguyên vẹn của các notebook đã dùng. Không cần chạy lại tất cả để mở demo. Mỗi notebook tự chứa mã cần thiết; upload một file lên Colab, không phải upload từng script Python.

| Notebook | Đã dùng để làm gì? | Khi nào cần chạy? |
|---|---|---|
| [M3Diff_LoRA_SCST_4_O.ipynb](M3Diff_LoRA_SCST_4_O.ipynb) | Huấn luyện hai nhánh A = LoRA + CE và B = LoRA + CE + SCST; tạo checkpoint B bước 128; đánh giá development 200 cặp. | Chỉ khi muốn xem hoặc tái chạy quá trình huấn luyện. Đã có checkpoint thì không cần train lại. |
| [M3Diff_SCST_Full1570_G100_B105_4_O.ipynb](M3Diff_SCST_Full1570_G100_B105_4_O.ipynb) | Đánh giá G và B trên 1.570 cặp test, tách Real/Render/All. Đây là notebook tạo bảng kết quả cuối hiện tại. | Khi muốn kiểm tra lại kết quả full test; kết quả sẵn có được giữ trên Drive. |
| [M3Diff_Demo_Web_7_O.ipynb](../M3Diff_Demo_Web_7_O.ipynb) | Nạp model gốc + adapter B128 và mở API để web laptop sử dụng. | Chỉ muốn mở demo: dùng file này, chạy 1 → 2 → 3 → 6 → 7. |

## Chia sẻ cho nhóm

Chia sẻ thư mục `demo/` và Drive `M3Diff_Storage/`. Mỗi người mount tài khoản Drive của mình; thư mục được chia sẻ cần truy cập được ở **MyDrive/M3Diff_Storage/**. Có thể thêm lối tắt vào MyDrive nếu Colab đọc được lối tắt đó, hoặc tạo bản sao các thư mục cần dùng tại đúng đường dẫn. Chỉ thấy thư mục trong “Được chia sẻ với tôi” chưa bảo đảm notebook đọc được.

Để demo dùng được checkpoint đã học, Drive cần có `M3Diff_checkpoint/` và thư mục thí nghiệm `MDP_Projector_v1/experiments/scst_lora_20261031/` (hoặc gói adapter đã xuất `demo_scst_B128/`). Ô 2 notebook demo tự xuất adapter từ checkpoint B128 nếu cần. Không bắt buộc tải checkpoint về laptop hay đưa trọng số lên GitHub.

Để chạy lại các thí nghiệm, chia sẻ nguyên `M3Diff_Storage/`, gồm dữ liệu/cache và kết quả đã lưu. Các notebook dùng đường dẫn cố định, protocol và checksum; không đổi tên thư mục/checkpoint. Có checkpoint và kết quả rồi thì các ô có thể dùng lại cache, không phải luôn bắt đầu một lượt train mới.

## Đọc kết quả nào?

- [CAI_TIEN_SCST_VA_KET_QUA.md](../CAI_TIEN_SCST_VA_KET_QUA.md) có phương pháp huấn luyện, mục 6.1 so với bài báo và mục 6.2 đối chiếu G/B thực chạy.
- File `notebook_manifest.json` lưu hash để kiểm tra bản sao khớp notebook trong `caitien/`.

Notebook confirmation 200 cặp và chẩn đoán G 10 cặp đã bỏ khỏi thư mục demo theo yêu cầu; bản gốc vẫn ở `caitien/` để lưu lịch sử kiểm chứng. Các notebook MDP/projector, trộn trọng số và bản full1570 cũ không nằm trong bộ chạy hiện tại. Chúng vẫn được giữ ở `caitien/` để lưu lịch sử, không xóa hoặc coi là đã tạo ra B128.
