## Hướng dẫn Cài đặt và Chạy chương trình

### Yêu cầu
- Python 3.8+
- `pip` và `venv` (thường đi kèm với Python)

### Các bước Cài đặt

1.  **Tải mã nguồn về:**
    Tải dự án từ GitHub về máy tính của bạn. Bạn có thể tải dưới dạng file `.zip` hoặc dùng `git`:
    ```bash
    git clone https://github.com/trhgbao/CSAI.git
    cd CSAI
    ```
2.  **Cài đặt các Thư viện cần thiết:**
    Nếu bạn dùng venv, nhóm cung cấp sẵn folder `venv` (!Lưu ý: môi trường này dùng Python 3.12). Tuy nhiên, nếu bạn muốn tạo lại môi trường mới thì hãy xóa thư mục `venv` cũ trước rồi thực hiện các bước sau:
    ```bash
    python3 -m venv venv
    pip install -r requirements.txt
    # Trên Windows:
    .\venv\bin\activate
    # Trên macOS/Linux:
    # source venv/bin/activate
    ```
    Nếu bạn dùng conda
    ```bash
    conda create -n csai python==3.12 -y
    conda activate csai
    ```

### Chạy Ứng dụng

```bash
python3 main.py
```

## Hướng dẫn Sử dụng các Chức năng

Giao diện chương trình được chia thành các khu vực được đánh số thứ tự.

### 1-5. Chức năng Cơ bản

1.  **Chọn Bài toán**: Click vào "Hàm Sphere" hoặc "Tô màu Đồ thị".
2.  **Chọn Thuật toán**: Chọn một thuật toán từ menu thả xuống. Danh sách sẽ tự động cập nhật theo bài toán đã chọn.
3.  **Điều chỉnh Tham số**: Tinh chỉnh các tham số đặc thù của thuật toán.
4.  **Tùy chọn Bài toán**: Cấu hình riêng cho bài toán (chọn số chiều cho Sphere, chọn file test case cho Tô màu).
5.  **Bắt đầu Trực quan hóa**:
    - **Bắt đầu**: Chạy thuật toán. Thông tin về thời gian và kết quả tốt nhất hiện tại sẽ được cập nhật "live" trên tiêu đề của biểu đồ chính.
    - **Tạm dừng / Tiếp tục**: Điều khiển animation.
    - **Đặt lại**: Xóa toàn bộ màn hình về trạng thái ban đầu.

### 6. Quản lý và So sánh các Lần chạy (Runs)

Sau khi một thuật toán chạy xong, bạn có thể lưu lại kết quả để so sánh với các lần chạy khác.

- **Lưu kết quả**: Nhấn nút này để lưu lại lịch sử hội tụ và thông số của lần chạy vừa hoàn thành. Một tên định danh (ví dụ: `PSO_D2_Iter100`) sẽ được thêm vào **"Danh sách các bản lưu"**.
- **Hiển thị so sánh hội tụ**: Chọn một hoặc nhiều bản lưu trong danh sách (giữ phím `Ctrl` và click chuột để chọn nhiều mục) và nhấn nút này. Biểu đồ hội tụ (bên dưới) sẽ vẽ tất cả các đường hội tụ của những lần chạy bạn đã chọn, giúp so sánh hiệu suất một cách trực quan.
- **Xóa bản lưu**: Các nút "Xóa bản lưu đã chọn" và "Xóa TẤT CẢ bản lưu" cho phép bạn quản lý danh sách các lần chạy đã lưu.

### 7. Trực quan hóa Song song

Đây là tính năng nâng cao cho phép bạn xem lại và so sánh quá trình hoạt động của hai lần chạy đã lưu một cách đồng thời.

- **Chọn Run 1 và Run 2**: Chọn hai bản lưu từ danh sách thả xuống. Danh sách này được lấy từ các kết quả bạn đã lưu ở mục 6.
- **Chạy minh họa song song**: Nhấn nút này. Hai ô "Kết quả Run 1" và "Kết quả Run 2" (nằm giữa hai biểu đồ chính) sẽ bắt đầu minh họa lại quá trình hội tụ của hai thuật toán đã chọn. (Tính năng này hoạt động tốt nhất với bài toán Sphere D=2).
- **Dừng / Xóa minh họa song song**: Các nút điều khiển cho phép bạn dừng hoặc xóa các animation đang chạy song song.
