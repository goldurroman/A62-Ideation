import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Analyse morphologique lésions",
    layout="wide"
)

MODEL_PATH = "yolov8m-seg-best.pt"
model = YOLO(MODEL_PATH)

st.title("🧠 Analyse de lésions cutanées - YOLOv8 Segmentation")

st.sidebar.header("Infos modèle")
st.sidebar.write("Task:", model.task)
st.sidebar.write("Classes:", model.names)

# ============================================================
# PARAMÈTRES
# ============================================================

conf_threshold = st.sidebar.slider(
    "Seuil de confiance",
    0.0, 1.0, 0.25, 0.01
)

imgsz = st.sidebar.selectbox(
    "Taille image",
    [256, 320, 640, 1024],
    index=2
)

# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Charger une image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# FONCTIONS
# ============================================================

def compute_morphology_scores(mask):

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    compactness = (4 * np.pi * area) / (perimeter**2 + 1e-6)

    H, W = mask.shape

    left = mask[:, :W//2].sum()
    right = mask[:, W//2:].sum()

    top = mask[:H//2, :].sum()
    bottom = mask[H//2:, :].sum()

    asymmetry = max(
        abs(left - right) / (left + right + 1e-6),
        abs(top - bottom) / (top + bottom + 1e-6)
    )

    (_, _), radius = cv2.minEnclosingCircle(cnt)
    diameter = radius * 2

    return {
        "compactness": float(compactness),
        "asymmetry": float(asymmetry),
        "diameter": float(diameter)
    }, cnt


def compute_global_suspicion_score(scores):

    comp = np.clip(scores["compactness"], 0, 1)
    asym = np.clip(scores["asymmetry"], 0, 1)
    diam = scores["diameter"]

    comp_score = (1 - comp) * 100
    asym_score = asym * 100

    diam_score = min(np.log1p(diam) / np.log1p(150), 1) * 100

    final_score = (
        0.40 * comp_score +
        0.40 * asym_score +
        0.20 * diam_score
    )

    return float(np.clip(final_score, 0, 100))

# ============================================================
# MAIN
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    st.image(img_rgb, caption="Image originale", use_container_width=True)

    if st.button("🚀 Lancer l'analyse"):

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        results = model.predict(
            source=img_bgr,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False
        )

        res = results[0]

        # ====================================================
        # YOLO RESULT
        # ====================================================

        result_img = res.plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Résultat YOLO", use_container_width=True)

        # ====================================================
        # MASK EXTRACTION
        # ====================================================

        if res.masks is None or len(res.masks.xy) == 0: # Use .xy to check for masks at original resolution
            st.error("Aucun masque détecté.")
            # Initialize binary_mask to an empty array of the correct size
            binary_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        else:
            # We select the mask with the highest confidence score.
            confs = res.boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)

            # Get the binary mask already scaled to the original image dimensions
            # res.masks.xy is a list of (H_orig, W_orig) binary masks.
            # The values are typically 0 or 1, but we apply the threshold for consistency.
            raw_binary_mask_orig_res = res.masks.xy[best_idx]
            binary_mask = (raw_binary_mask_orig_res >= 0.20).astype(np.uint8) # Apply threshold from context

            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

            st.subheader("Masque final")
            st.image(binary_mask * 255, use_container_width=True)

            # ====================================================
            # OVERLAY
            # ====================================================

            # binary_mask is now already at the correct img_rgb.shape[:2] resolution.
            # No need for cv2.resize here.
            overlay_mask_color = np.zeros_like(img_rgb)
            overlay_mask_color[..., 0] = binary_mask * 255 # Apply red color to the mask

            final_overlay = cv2.addWeighted(
                img_rgb, 1, overlay_mask_color, 0.4, 0
            )

            st.image(final_overlay, caption="Overlay", use_container_width=True)

            # ====================================================
            # MORPHOLOGICAL ANALYSIS
            # ====================================================

            scores_morpho, cnt = compute_morphology_scores(binary_mask)

            if scores_morpho is not None:

                suspicion_score = compute_global_suspicion_score(scores_morpho)

                st.subheader("Analyse morphologique")

                st.metric(
                    "Score global de suspicion",
                    f"{suspicion_score:.1f}/100"
                )

                st.progress(suspicion_score / 100)

                # ====================================================
                # UNIQUE LOGIC (CORRIGÉE)
                # ====================================================

                if suspicion_score >= 70:
                    st.error("Niveau de complexité : ÉLEVÉ")

                elif suspicion_score >= 40:
                    st.warning("Niveau de complexité : MODÉRÉ")

                else:
                    st.success("Niveau de complexité : FAIBLE")

                st.info(
                    "Score basé uniquement sur la morphologie (forme, asymétrie, taille). "
                    "Ce résultat n'est PAS un diagnostic médical."
                )
 