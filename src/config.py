"""Cấu hình dùng chung cho cả nhóm (Người 1 -> Người 5).

Người 2 (Pre-processor) định nghĩa các hằng số liên quan đến kích thước ảnh
và số kênh màu để các thành viên khác (đặc biệt Người 3 - Model, Người 5 - UI)
import dùng cho đồng nhất, tránh lệch shape khi tích hợp.
"""

# Kích thước ảnh sau khi resize (GTSRB thường dùng 30x30)
IMG_HEIGHT = 30
IMG_WIDTH = 30
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)  # OpenCV nhận (width, height)

# Nếu True: ảnh xám -> kênh = 1 ; Nếu False: ảnh màu RGB -> kênh = 3
USE_GRAYSCALE = False
NUM_CHANNELS = 1 if USE_GRAYSCALE else 3

# input_shape để Người 3 dùng cho lớp Conv2D đầu tiên
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

# Số lớp biển báo của bộ dữ liệu GTSRB
NUM_CLASSES = 43
