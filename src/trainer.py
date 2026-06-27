import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import hàm build_model từ file cnn_model.py cùng thư mục src
from cnn_model import build_model 

def create_mock_data(num_samples=1000, input_shape=(32, 32, 3), num_classes=43):
    """
    Tạo dữ liệu giả lập (Mock Data) khớp với cấu trúc hình ảnh đầu vào (32x32x3)
    và số lượng nhóm biển báo (43 classes) để test code độc lập.
    Khi Người 1 và Người 2 làm xong, đoạn này sẽ được thay bằng dữ liệu thật.
    """
    print("=== [Người 4] Đang khởi tạo dữ liệu giả lập để kiểm thử ===")
    X_mock = np.random.rand(num_samples, *input_shape).astype('float32')
    
    # GTSRB có dữ liệu nhãn dạng số nguyên từ 0 đến 42
    y_mock = np.random.randint(0, num_classes, size=(num_samples,))
    
    # Chia dữ liệu thành các tập Train, Validation, Test
    X_train, X_val = X_mock[:800], X_mock[800:]
    y_train, y_val = y_mock[:800], y_mock[800:]
    X_test, y_test = X_mock[:100], y_mock[:100] 
    
    return X_train, X_val, y_train, y_val, X_test, y_test

def plot_training_history(history):
    """
    Nhiệm vụ: Vẽ đồ thị trực quan hóa quá trình học tập (Loss và Accuracy).
    """
    print("=== [Người 4] Đang vẽ đồ thị lịch sử huấn luyện ===")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    
    # 1. Đồ thị Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    # 2. Đồ thị Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    
    # SỬA TẠI ĐÂY: Lưu đồ thị ngay tại thư mục gốc của dự án, không lùi ra ngoài nữa
    plt.savefig("training_history.png", dpi=300)
    print("-> Đã lưu đồ thị học tập tại: training_history.png")
    plt.show()

def evaluate_and_plot_confusion_matrix(model, X_test, y_test, num_classes=43):
    """
    Nhiệm vụ: Tính toán ma trận nhầm lẫn (Confusion Matrix).
    """
    print("=== [Người 4] Đang đánh giá mô hình trên tập kiểm thử (Test Set) ===")
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1) 
    
    # Tính toán Confusion Matrix bằng thư viện sklearn [cite: 130]
    cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
    
    # Vẽ biểu đồ nhiệt (Heatmap) thể hiện ma trận
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label (Nhãn dự đoán)')
    plt.ylabel('True Label (Nhãn thực tế)')
    plt.title('Traffic Sign Classification - Confusion Matrix')
    
    # SỬA TẠI ĐÂY: Lưu ma trận nhầm lẫn ngay tại thư mục gốc của dự án
    plt.savefig("confusion_matrix.png", dpi=300)
    print("-> Đã lưu ma trận nhầm lẫn tại: confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    # SỬA TẠI ĐÂY: Tạo thư mục models bên trong thư mục gốc dự án nếu chưa có
    os.makedirs("models", exist_ok=True)

    # 1. Khởi tạo tập dữ liệu giả lập ban đầu
    X_train, X_val, y_train, y_val, X_test, y_test = create_mock_data()

    # 2. Chuyển đổi nhãn sang One-hot mã hóa để khớp hàm lỗi mạng CNN
    y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=43)
    y_val_encoded = tf.keras.utils.to_categorical(y_val, num_classes=43)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=43)

    model = build_model(input_shape=(32, 32, 3), num_classes=43, model_type="improved")
    model.summary()
    
    # 3. Tiến hành quá trình huấn luyện mô hình [cite: 127]
    print("\n=== [Người 4] Bắt đầu quá trình huấn luyện mạng Neural ===")
    history = model.fit(
        X_train, y_train_encoded,
        epochs=5, 
        batch_size=32,
        validation_data=(X_val, y_val_encoded)
    )

    # SỬA TẠI ĐÂY: Lưu file mô hình vào đúng thư mục models nằm ĐẰNG TRONG dự án
    model_save_path = "models/traffic_sign_model.h5"
    model.save(model_save_path)
    print(f"\n-> Đã xuất và đóng gói file mô hình thành công tại: {model_save_path}")

    # 5. Thực thi các tác vụ vẽ báo cáo và phân tích sai số [cite: 128, 129]
    plot_training_history(history)
    evaluate_and_plot_confusion_matrix(model, X_test, y_test)