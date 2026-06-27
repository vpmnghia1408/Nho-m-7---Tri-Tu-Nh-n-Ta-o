import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🤖",
    layout="centered",
)


st.title("🤖 AI Image Classifier")
st.markdown("Tải ảnh lên để nhận kết quả dự đoán từ mô hình AI đã huấn luyện.")
st.divider()


st.sidebar.header("⚙️ Cài đặt mô hình")

model_type = st.sidebar.radio(
    "Loại mô hình:",
    options=["Keras (.h5)", "PyTorch (.pth)"],
    index=0,
)

model_path = st.sidebar.text_input(
    "Đường dẫn file mô hình:",
    value="model.h5" if model_type == "Keras (.h5)" else "model.pth",
    help="Nhập đường dẫn tuyệt đối hoặc tương đối tới file mô hình.",
)

# Nhãn phân loại – người dùng có thể tuỳ chỉnh
default_labels = "Mèo, Chó, Chim, Cá, Khác"
labels_input = st.sidebar.text_area(
    "Nhãn phân loại (cách nhau bằng dấu phẩy):",
    value=default_labels,
    help="Thứ tự nhãn phải khớp với output của mô hình.",
)
CLASS_NAMES = [lbl.strip() for lbl in labels_input.split(",") if lbl.strip()]

# Kích thước ảnh đầu vào
img_size = st.sidebar.slider("Kích thước ảnh đầu vào (px):", 32, 512, 224, step=32)

st.sidebar.divider()
st.sidebar.info(
    "💡 **Hướng dẫn:**\n"
    "1. Chọn loại mô hình.\n"
    "2. Nhập đường dẫn file `.h5` hoặc `.pth`.\n"
    "3. Điền đúng tên nhãn.\n"
    "4. Upload ảnh và xem kết quả!"
)

# HÀM LOAD MÔ HÌNH (có cache để không load lại mỗi lần)

@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_keras_model(path: str):
    """Load mô hình Keras / TensorFlow từ file .h5"""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        model = keras.models.load_model(path)
        return model, None
    except ImportError:
        return None, "❌ Chưa cài TensorFlow. Chạy: `pip install tensorflow`"
    except Exception as e:
        return None, f"❌ Không thể load mô hình Keras: {e}"


@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_pytorch_model(path: str):
    """Load mô hình PyTorch từ file .pth"""
    try:
        import torch
        # Dùng weights_only=False để tương thích nhiều loại checkpoint
        model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        model.eval()
        return model, None
    except ImportError:
        return None, "❌ Chưa cài PyTorch. Chạy: `pip install torch`"
    except Exception as e:
        return None, f"❌ Không thể load mô hình PyTorch: {e}"



# HÀM TIỀN XỬ LÝ ẢNH

def preprocess_image_keras(image: Image.Image, size: int) -> np.ndarray:
    """Chuyển PIL Image → numpy array chuẩn cho Keras"""
    img = image.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0   # Chuẩn hoá [0, 1]
    arr = np.expand_dims(arr, axis=0)                 # Thêm batch dimension
    return arr


def preprocess_image_pytorch(image: Image.Image, size: int):
    """Chuyển PIL Image → tensor chuẩn cho PyTorch"""
    try:
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # Chuẩn ImageNet
                std=[0.229, 0.224, 0.225],
            ),
        ])
        tensor = transform(image.convert("RGB"))
        tensor = tensor.unsqueeze(0)  # Thêm batch dimension
        return tensor, None
    except ImportError:
        return None, "❌ Chưa cài torchvision. Chạy: `pip install torchvision`"


# HÀM DỰ ĐOÁN

def predict_keras(model, image: Image.Image, size: int, class_names: list):
    """Dự đoán bằng Keras model"""
    arr = preprocess_image_keras(image, size)
    preds = model.predict(arr, verbose=0)[0]           # Shape: (num_classes,)
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    label = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"

    # Top-3 kết quả
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [
        (class_names[i] if i < len(class_names) else f"Class {i}", float(preds[i]) * 100)
        for i in top3_idx
    ]
    return label, confidence, top3, None


def predict_pytorch(model, image: Image.Image, size: int, class_names: list):
    """Dự đoán bằng PyTorch model"""
    try:
        import torch
        import torch.nn.functional as F

        tensor, err = preprocess_image_pytorch(image, size)
        if err:
            return None, None, None, err

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0]   # Shape: (num_classes,)

        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item()) * 100

        label = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"

        top3_idx = torch.topk(probs, min(3, len(probs))).indices.tolist()
        top3 = [
            (class_names[i] if i < len(class_names) else f"Class {i}", float(probs[i].item()) * 100)
            for i in top3_idx
        ]
        return label, confidence, top3, None
    except Exception as e:
        return None, None, None, f"❌ Lỗi khi dự đoán: {e}"


st.subheader("📤 Upload ảnh")
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh từ máy tính (JPG, PNG, BMP, WEBP):",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="visible",
)

if uploaded_file is not None:
    # Hiển thị ảnh gốc
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="📷 Ảnh đã tải lên", use_container_width=True)
        st.caption(f"Kích thước gốc: {image.width} × {image.height} px")

    with col2:
        st.markdown("### 🔍 Kết quả dự đoán")

        # Kiểm tra file mô hình tồn tại
        if not os.path.exists(model_path):
            st.error(
                f"⚠️ Không tìm thấy file mô hình tại:\n`{model_path}`\n\n"
                "Vui lòng kiểm tra lại đường dẫn ở thanh bên trái."
            )

            # Demo mode – hiển thị kết quả giả để test giao diện
            st.info("🎭 **Demo Mode** – Đang hiển thị kết quả mẫu vì chưa có mô hình thật.")
            demo_label = CLASS_NAMES[0] if CLASS_NAMES else "Demo Label"
            demo_conf = 87.5
            st.success(f"**{demo_label}**")
            st.metric(label="Confidence", value=f"{demo_conf:.2f}%")
            st.progress(int(demo_conf))

            st.markdown("**📊 Top-3 dự đoán:**")
            demo_top3 = [
                (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}", round(87.5 - i * 20, 1))
                for i in range(min(3, len(CLASS_NAMES)))
            ]
            for rank, (lbl, conf) in enumerate(demo_top3, 1):
                st.write(f"{rank}. **{lbl}** — `{conf:.1f}%`")
                st.progress(max(0, int(conf)))

        else:
            # ── Load mô hình thật ──
            with st.spinner("Đang phân tích ảnh…"):
                if model_type == "Keras (.h5)":
                    model, err = load_keras_model(model_path)
                    if err:
                        st.error(err)
                        model = None
                else:
                    model, err = load_pytorch_model(model_path)
                    if err:
                        st.error(err)
                        model = None

            # ── Dự đoán ──
            if model is not None:
                if model_type == "Keras (.h5)":
                    label, confidence, top3, err = predict_keras(model, image, img_size, CLASS_NAMES)
                else:
                    label, confidence, top3, err = predict_pytorch(model, image, img_size, CLASS_NAMES)

                if err:
                    st.error(err)
                else:
                    # Kết quả chính
                    st.success(f"**{label}**")
                    st.metric(label="Confidence", value=f"{confidence:.2f}%")
                    st.progress(int(confidence))

                    st.markdown("**📊 Top-3 dự đoán:**")
                    for rank, (lbl, conf) in enumerate(top3, 1):
                        st.write(f"{rank}. **{lbl}** — `{conf:.1f}%`")
                        st.progress(max(0, int(conf)))

else:
    # Placeholder khi chưa có ảnh
    st.info("👆 Hãy upload một ảnh để bắt đầu dự đoán.")
    st.markdown(
        """
        **Hướng dẫn nhanh:**
        1. Cài thư viện: `pip install streamlit pillow tensorflow` *(hoặc `torch torchvision`)*
        2. Đặt file mô hình (`.h5` / `.pth`) vào thư mục dự án.
        3. Nhập đường dẫn mô hình ở thanh bên trái.
        4. Upload ảnh và xem kết quả ngay lập tức!
        """
    )

# FOOTER

st.divider()
st.caption("👨‍💻 Người 5 – Fullstack AI Developer | Nhóm 7 – Trí Tuệ Nhân Tạo")
