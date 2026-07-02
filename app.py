import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# =========================
# MODEL LOAD (FORCÉ)
# =========================
MODEL_PATH = "C:/Users/sarah/streamlit_app/best.pt"
model = YOLO(MODEL_PATH)

st.title("🧠 YOLOv8 DETECTION - DEBUG MODE")

# =========================
# DEBUG MODEL INFO
# =========================
st.sidebar.header("🔍 Debug Model")
st.sidebar.write("Model loaded:", MODEL_PATH)
st.sidebar.write("Task:", model.task)
st.sidebar.write("Classes:", model.names)

# =========================
# PARAMS
# =========================
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.selectbox("Image size", [256, 320, 640, 1024], index=2)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])

def load_image(file):
    img = Image.open(file).convert("RGB")
    return np.array(img)

# =========================
# MAIN
# =========================
if uploaded_file is not None:

    img = load_image(uploaded_file)

    st.image(img, caption="Input Image", use_container_width=True)

    st.write("Image shape:", img.shape)

    if st.button("🚀 RUN DETECTION"):

        with st.spinner("Running YOLO inference..."):

            # 🔥 IMPORTANT: convert RGB → BGR (YOLO friendly)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            results = model.predict(
                source=img_bgr,
                conf=conf_threshold,
                imgsz=imgsz,
                verbose=True
            )

            res = results[0]

            # =========================
            # DEBUG
            # =========================
            st.write("🔎 Number of boxes:", len(res.boxes))

            if len(res.boxes) > 0:
                st.write("Conf:", res.boxes.conf.cpu().numpy())
                st.write("Classes:", res.boxes.cls.cpu().numpy())

                st.write("Max confidence:", float(res.boxes.conf.max()))

            else:
                st.warning("❌ Aucune détection")

            # =========================
            # RESULT IMAGE
            # =========================
            result_img = res.plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            st.image(result_img, caption="Detection result", use_container_width=True)