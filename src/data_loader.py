import os
import cv2
import numpy as np
import pandas as pd

# =========================
# CẤU HÌNH
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "dataset", "raw", "gtsrb")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

IMG_SIZE = 32


# =========================
# HÀM ĐỌC ẢNH TỪ FILE CSV ĐÃ CHIA
# =========================

def load_images_from_csv(csv_path, img_size=32, limit=None):
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)

    images = []
    labels = []

    for index, row in df.iterrows():
        image_path = os.path.join(DATA_DIR, row["Path"])
        label = int(row["ClassId"])

        image = cv2.imread(image_path)

        if image is None:
            print("Không đọc được ảnh:", image_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size))
        image = image.astype("float32") / 255.0

        images.append(image)
        labels.append(label)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return X, y


# =========================
# CHẠY THỬ
# =========================

if __name__ == "__main__":
    train_csv = os.path.join(PROCESSED_DIR, "train_split.csv")
    val_csv = os.path.join(PROCESSED_DIR, "val_split.csv")
    test_csv = os.path.join(PROCESSED_DIR, "test_split.csv")

    print("Đang đọc thử dữ liệu Train...")

    X_train_sample, y_train_sample = load_images_from_csv(
        train_csv,
        IMG_SIZE,
        limit=1000
    )

    print("X_train_sample:", X_train_sample.shape)
    print("y_train_sample:", y_train_sample.shape)
    print("Danh sách nhãn:", np.unique(y_train_sample))