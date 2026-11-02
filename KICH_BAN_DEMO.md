# Kịch bản trình diễn khoảng 5 phút

Chuẩn bị trước giờ học: mở `M3Diff_Demo_Web_7_O.ipynb` trên Colab, bật GPU và chạy 1 → 2 → 3 → 6 → 7. Mở web laptop và nhập file kết nối do ô 7 tải về. Chạy thử một cặp ảnh để tránh phải chờ cài thư viện/tải checkpoint trước mặt cô. Chuẩn bị 3–5 cặp ảnh, ít nhất một cặp có thay đổi rõ và một cặp cùng ảnh.

1. **Giới thiệu (30 giây).** “Nhóm đã fine-tune M³Diff bằng LoRA kết hợp SCST và đưa checkpoint B128 vào demo. Ứng dụng có so sánh toàn ảnh, chọn vùng, lưu kết quả và xuất báo cáo. Bảng full1570 và cách huấn luyện nằm trong tài liệu kèm repo.”
2. **So sánh toàn ảnh (1 phút).** Chọn hai ảnh trước/sau có thay đổi dễ thấy. Bấm so sánh, đọc caption và chỉ vào ảnh để kiểm chứng. Cho xem thời gian model và thời gian tổng có cả mạng.
3. **Chọn vùng (1–2 phút).** Khoanh vùng có chi tiết cần chú ý ở ảnh trước. Chép sang ảnh sau nếu hai ảnh cùng góc chụp, hoặc chọn vùng tương ứng riêng. Cho cô xem hai ảnh cắt trước khi bấm so sánh vùng. Đối chiếu với kết quả toàn ảnh. Nếu model sai, nêu đúng sai thay vì chọn lọc che lỗi.
4. **Lưu và báo cáo (1 phút).** Lưu kết quả toàn ảnh/vùng. Mở tab lịch sử, tải HTML hoặc ZIP. Báo cáo giữ ảnh, vùng chọn, caption, thời gian và mã phiên; có thể mở lại khi ngắt GPU.
5. **Nêu kết quả và giới hạn (30 giây).** “Cấu hình B tăng BLEU-4, ROUGE-L và CIDEr tổng thể trên bảng full1570; METEOR giảm nhẹ và CIDEr Render giảm. G/B khác cấu hình sinh câu nên chưa tách riêng mức tăng do huấn luyện. Vùng vẫn do người dùng chọn; model có thể mô tả sai.”

## Bộ tình huống nên kiểm tra trước

| Tình huống | Thao tác | Điều cần quan sát |
|---|---|---|
| Thay đổi rõ | So sánh toàn ảnh cặp 1 | Caption có đúng thay đổi nhìn thấy không |
| Chi tiết nhỏ | Toàn ảnh rồi hai vùng tương ứng | Ảnh cắt có đúng chi tiết, model có mô tả khác không |
| Cùng ảnh | Chọn cùng một ảnh trước/sau | Model có bịa thay đổi không; ghi nhận trung thực |
| Đổi cặp | Sau khi có kết quả, thay ảnh sau | Kết quả cũ biến mất, không tự gọi GPU |
| Mất kết nối | Đóng riêng web Gradio sau khi đã lưu | Lượt mới báo lỗi, ảnh vẫn giữ; lịch sử/báo cáo vẫn mở |

Đây là kiểm tra chức năng và minh họa định tính; không suy ra điểm benchmark từ vài cặp demo. Để xác nhận web dùng GPU thật, bấm so sánh một lần và xem mã model giống mã trong Colab, số lần nạp runner không tăng; lưu báo cáo lần đó.
