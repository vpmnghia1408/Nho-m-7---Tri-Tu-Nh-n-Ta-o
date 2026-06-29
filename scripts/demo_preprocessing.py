"""Script demo cho phần Người 2 - Data Pre-processor.

Mục đích: kiểm chứng nhanh module tiền xử lý chạy đúng MÀ KHÔNG cần dataset thật.
Script tự tạo vài ảnh giả ngẫu nhiên (giả lập output của Người 1) rồi chạy
toàn bộ pipeline: resize -> grayscale (tùy chọn) -> normalize -> augment.

Cách chạy (từ thư mục gốc dự án):
    python -m scripts.demo_preprocessing
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.config import INPUT_SHAPE, IMG_SIZE
from src.preprocessing import (
    preprocess_image,
    preprocess_batch,
    augment_image,
    create_augmenter,
)


def make_fake_raw_images(n: int = 5) -> np.ndarray:
    """Tạo n ảnh màu thô kích thước bất kỳ (giả lập dữ liệu từ Người 1)."""
    rng = np.random.default_rng(42)
    return [
        rng.integers(0, 256, size=(rng.integers(40, 80), rng.integers(40, 80), 3),
                     dtype=np.uint8)
        for _ in range(n)
    ]


def main() -> None:
    print("=== DEMO TIỀN XỬ LÝ - NGƯỜI 2 ===\n")

    raw_images = make_fake_raw_images(5)
    print(f"[1] Số ảnh thô: {len(raw_images)} | kích thước ảnh đầu: {raw_images[0].shape}")

    # Xử lý 1 ảnh
    one = preprocess_image(raw_images[0])
    print(f"[2] preprocess_image -> shape {one.shape}, "
          f"min={one.min():.3f}, max={one.max():.3f}, dtype={one.dtype}")

    # Xử lý cả batch
    batch = preprocess_batch(raw_images)
    print(f"[3] preprocess_batch -> shape {batch.shape} (kỳ vọng (5, {IMG_SIZE[1]}, {IMG_SIZE[0]}, ...))")

    # Kiểm tra khớp INPUT_SHAPE để Người 3 dùng đúng
    assert batch.shape[1:] == INPUT_SHAPE, (
        f"Shape không khớp INPUT_SHAPE! {batch.shape[1:]} != {INPUT_SHAPE}"
    )
    print(f"[4] Shape mỗi ảnh KHỚP INPUT_SHAPE {INPUT_SHAPE} -> sẵn sàng cho model CNN")

    # Kiểm tra normalize đúng đoạn [0, 1]
    assert 0.0 <= batch.min() and batch.max() <= 1.0, "Pixel chưa nằm trong [0, 1]!"
    print("[5] Pixel đã chuẩn hóa trong [0, 1] -> OK")

    # Demo augmentation thủ công
    aug = augment_image(raw_images[0], seed=0)
    print(f"[6] augment_image (xoay + zoom) -> shape {aug.shape} -> OK")

    # Demo tạo augmenter cho Người 4 (chỉ tạo, không train)
    try:
        gen = create_augmenter()
        print(f"[7] create_augmenter() -> {type(gen).__name__} (sẵn sàng cho model.fit)")
    except ImportError:
        print("[7] (Bỏ qua) Chưa cài tensorflow -> create_augmenter cần Keras")

    print("\n=== TẤT CẢ BƯỚC TIỀN XỬ LÝ CHẠY THÀNH CÔNG ===")


if __name__ == "__main__":
    main()
