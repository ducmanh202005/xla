# Chia sẻ notebook và Drive cho nhóm

## Chia sẻ những gì đã chạy

1. Đưa thư mục `demo/` lên repo nhóm, gồm [các notebook cải tiến](notebooks/README.md), tài liệu và kết quả.
2. Chia sẻ Drive **M3Diff_Storage/** đang chứa model gốc và thí nghiệm đã học. Thành viên cần truy cập được tại **MyDrive/M3Diff_Storage/** trong tài khoản họ mount vào Colab. Thư mục nằm trong “Được chia sẻ với tôi” chưa chắc xuất hiện ở đường dẫn này; thêm lối tắt vào MyDrive nếu Colab đọc được hoặc sao chép các thư mục cần thiết về đúng vị trí.
3. Chỉ chạy demo: mở `M3Diff_Demo_Web_7_O.ipynb`, chọn A100, chạy **1 → 2 → 3 → 6 → 7**. Ô 2 tự dùng gói adapter đã có hoặc xuất đúng B128 từ thư mục train được chia sẻ; không huấn luyện lại.
4. Trên laptop mở `CHAY_DEMO.cmd`, nhập file kết nối mới của ô 7. Không cần tải model về laptop.

**Đổi máy tính vẫn chạy được** vì phần model nằm trên GPU Colab, không chạy trên máy cá nhân. Dùng cùng tài khoản Drive thì đường dẫn giữ nguyên; dùng tài khoản thành viên thì thư mục chia sẻ phải đọc được tại đường dẫn notebook yêu cầu. Khi ô 2 tạo `demo_scst_B128/` lần đầu, cần quyền ghi vào thư mục lưu trữ. Nếu adapter đã có hợp lệ, notebook dùng lại nó. Không cần gửi thêm checkpoint B riêng.

Để demo, giữ model gốc `M3Diff_checkpoint/` và thí nghiệm `MDP_Projector_v1/experiments/scst_lora_20261031/` hoặc adapter đã xuất `demo_scst_B128/`. Để chạy lại các notebook thí nghiệm như người tạo, chia sẻ nguyên `M3Diff_Storage/`, gồm dữ liệu/cache/kết quả. Các notebook trong `notebooks/` là bản sao đúng file đã dùng; mỗi file có sẵn toàn bộ mã hỗ trợ.

Không cần chạy lại notebook huấn luyện để mở web. Nếu thành viên muốn huấn luyện lại, nên tạo bản sao Drive riêng để tránh nhiều người cùng ghi vào checkpoint chung. Checkpoint/kết quả đã có sẽ được notebook dùng lại theo điều kiện kiểm tra; không tự xóa chúng để ép chạy mới.

## Thư mục nào trên Drive tạo ra kết quả này?

Trong **MyDrive/M3Diff_Storage/**:

| Thư mục/file | Vai trò | Cần giữ/lấy gì? |
|---|---|---|
| `MDP_Projector_v1/experiments/scst_lora_20261031/checkpoints/B/` | Checkpoint B được huấn luyện | Giữ toàn bộ để sao lưu. `commit.json` chỉ rõ slot0.pt hay slot1.pt hiện hành; không tự đoán slot. |
| `MDP_Projector_v1/experiments/scst_lora_20261031/protocol.json` và `weights_B.json` | Danh tính cấu hình/trọng số | Giữ cùng checkpoint; ô 2 dùng để kiểm tra. |
| `MDP_Projector_v1/experiments/scst_lora_20261031/full_test_scst_G100_B105_v1/` | Caption, bảng điểm, log full1570 | Đã có trong ZIP báo cáo bạn gửi; không phải checkpoint. Giữ làm bằng chứng. |
| `demo_scst_B128/` | Adapter suy luận do ô 2 tự tạo từ checkpoint B128 | Dùng ngay trên Drive chung; không phải tải về hay gửi riêng. |
| `M3Diff_checkpoint/` | Model gốc gồm bốn shard, tokenizer/config | Bắt buộc cho suy luận. Đã có thì giữ nguyên, không tải lại để đưa vào repo. |

Checkpoint được chọn: nhánh **B**, bước **128**, SHA256 file checkpoint huấn luyện:
`20d99f77346716fef35d5b369f6eee83458e8c854566d7bd3a557e5b2ba53a83`.

Hash của file adapter xuất ra sẽ khác hash checkpoint huấn luyện vì đã bỏ optimizer/RNG. `adapter_manifest.json` ghi cả nguồn checkpoint và hash file adapter mới.

Muốn sao lưu để **train tiếp**, giữ toàn bộ thư mục `scst_lora_20261031/`; chỉ hai file của gói demo không chứa trạng thái optimizer để resume train.

### Chưa thấy thư mục demo_scst_B128 trên Drive?

Đây là thư mục xuất riêng cho demo, không phải tên thư mục checkpoint đã huấn luyện. **Bạn không cần tự tạo hoặc huấn luyện lại.** Khi chạy ô 2, notebook kiểm tra checkpoint B128 trong:

`MyDrive/M3Diff_Storage/MDP_Projector_v1/experiments/scst_lora_20261031/`

Nếu đúng checkpoint, ô 2 tự tạo **MyDrive/M3Diff_Storage/demo_scst_B128/** chứa `adapter_model.pt` và `adapter_manifest.json`. Sau khi ô 2 báo **CHECKPOINT B128 ĐẠT**, chạy tiếp ô 3 → 6 → 7. Nếu thiếu checkpoint nguồn, notebook sẽ báo lỗi; tạo thư mục rỗng không thay thế được trọng số.

## Đưa gì lên GitHub?

Đưa mã, notebook, tài liệu, bảng điểm, source snapshot và LICENSE trong `demo/`. Chia sẻ model/checkpoint qua Drive là đủ; không cần đưa bốn shard model gốc hay slot huấn luyện lên Git. Không đưa file kết nối có mật khẩu hoặc lịch sử ảnh cá nhân lên repo. Chưa thực hiện push hoặc gửi thông tin cho thành viên nào.
