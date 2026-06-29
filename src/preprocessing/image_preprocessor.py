"""Module tiền xử lý ảnh - Công việc của Người 2 (Data Pre-processor).

Nhiệm vụ:
    - resize ảnh về cùng kích thước (mặc định 30x30)
    - chuyển ảnh màu -> ảnh xám (tùy chọn)
    - chuẩn hóa pixel từ [0, 255] về [0, 1]

Module này nhận dữ liệu thô (numpy array hoặc đường dẫn file) và trả về
mảng đã sẵn sàng đưa vào mô hình CNN của Người 3.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import IMG_SIZE, USE_GRAYSCALE


def load_image(path: str) -> np.ndarray:
    """Đọc 1 ảnh từ đường dẫn, trả về mảng RGB (H, W, 3) kiểu uint8.

    Dùng khi cần đọc ảnh trực tiếp từ file (ví dụ Người 5 - giao diện).
    OpenCV đọc ảnh ở thứ tự BGR nên ta đổi sang RGB cho thống nhất.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh tại: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Resize ảnh về kích thước cố định (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh màu sang ảnh xám, giữ chiều kênh -> (H, W, 1)."""
    if image.ndim == 3 and image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return np.expand_dims(gray, axis=-1)


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """Chuẩn hóa giá trị pixel từ [0, 255] về [0, 1], kiểu float32."""
    return image.astype("float32") / 255.0


def preprocess_image(
    image: "np.ndarray | str",
    size: tuple[int, int] = IMG_SIZE,
    grayscale: bool = USE_GRAYSCALE,
    normalize: bool = True,
) -> np.ndarray:
    """Tiền xử lý hoàn chỉnh cho MỘT ảnh.

    Args:
        image: numpy array (H, W, 3) hoặc đường dẫn tới file ảnh.
        size: kích thước đích (width, height).
        grayscale: True -> chuyển ảnh xám (H, W, 1).
        normalize: True -> đưa pixel về [0, 1].

    Returns:
        Ảnh đã xử lý, shape (H, W, C).
    """
    if isinstance(image, str):
        image = load_image(image)

    image = resize_image(image, size)

    if grayscale:
        image = to_grayscale(image)

    if normalize:
        image = normalize_pixels(image)

    return image


def preprocess_batch(
    images: "np.ndarray | list",
    size: tuple[int, int] = IMG_SIZE,
    grayscale: bool = USE_GRAYSCALE,
    normalize: bool = True,
) -> np.ndarray:
    """Tiền xử lý cho NHIỀU ảnh (nhận output từ Người 1 - Data Loader).

    Args:
        images: list ảnh hoặc mảng numpy (N, H, W, 3).
        size, grayscale, normalize: giống preprocess_image.

    Returns:
        Mảng numpy (N, H, W, C) đã tiền xử lý, sẵn sàng cho model.fit().
    """
    processed = [
        preprocess_image(img, size=size, grayscale=grayscale, normalize=normalize)
        for img in images
    ]
    return np.array(processed, dtype="float32")
