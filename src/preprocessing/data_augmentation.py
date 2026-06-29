"""Module Data Augmentation - Công việc nâng cao của Người 2.

Tạo thêm dữ liệu giả lập (xoay, phóng to/thu nhỏ, dịch chuyển) giúp mô hình
học tốt hơn và giảm overfitting.

Cung cấp 2 cách dùng:
    1. create_augmenter(): trả về ImageDataGenerator của Keras để Người 4
       đưa trực tiếp vào model.fit() khi huấn luyện.
    2. augment_image(): xoay + zoom thủ công bằng OpenCV cho 1 ảnh
       (dùng để xem trước/demo hiệu ứng augmentation).

LƯU Ý: KHÔNG lật ảnh trái-phải (horizontal flip) vì biển báo giao thông
bị lật có thể đổi ý nghĩa (ví dụ rẽ trái -> rẽ phải).
"""

from __future__ import annotations

import cv2
import numpy as np


def create_augmenter(
    rotation_range: int = 10,
    zoom_range: float = 0.15,
    width_shift_range: float = 0.1,
    height_shift_range: float = 0.1,
):
    """Tạo ImageDataGenerator để Người 4 dùng khi train.

    Cách dùng (cho Người 4):
        aug = create_augmenter()
        model.fit(aug.flow(X_train, y_train, batch_size=32),
                  validation_data=(X_val, y_val), epochs=20)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=False,
        fill_mode="nearest",
    )


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Xoay ảnh quanh tâm một góc (độ)."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def zoom_image(image: np.ndarray, factor: float) -> np.ndarray:
    """Phóng to (factor > 1) hoặc thu nhỏ (factor < 1) ảnh, giữ nguyên kích thước."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle=0, scale=factor)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment_image(
    image: np.ndarray,
    max_angle: float = 10.0,
    zoom_low: float = 0.9,
    zoom_high: float = 1.1,
    seed: "int | None" = None,
) -> np.ndarray:
    """Sinh 1 phiên bản augment ngẫu nhiên (xoay + zoom) cho 1 ảnh.

    Dùng cho mục đích demo/xem trước hiệu ứng augmentation.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-max_angle, max_angle)
    factor = rng.uniform(zoom_low, zoom_high)

    result = rotate_image(image, angle)
    result = zoom_image(result, factor)
    return result
