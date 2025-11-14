## Hướng dẫn Cài đặt và Chạy chương trình

### Yêu cầu
- Python 3.12
- `pip` và `venv` (thường đi kèm với Python) hoặc conda

### Các bước Cài đặt

1.  **Tải source code:**
    ```bash
    git clone https://github.com/trhgbao/CSAI.git
    cd CSAI
    ```
2. **Test cases:**
   Các testcase lưu trong mục `data` dưới dạng file `.txt`. Nếu bạn thêm testcase mới đảm bảo testcase của bạn đúng định dạng và 1-based:
   ```bash
   vertices edges
   1 2
   1 3
   ```
4.  **Cài đặt các Thư viện cần thiết:**
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

### Chạy thuật toán GC
Thay đổi config file trong `config/gc/<algo>.yaml`
```bash
python3 gc.py --algo <algo>
# python3 gc.py --algo aco
```


### Chạy thuật toán Sphere Function
Thay đổi config file trong `config/sphere/<algo>.yaml`
```bash
python3 sphere.py --algo <algo>
# python3 sphere.py --algo aco
```
