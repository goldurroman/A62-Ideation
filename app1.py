import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Analyse de lésions cutanées",
    layout="wide"
)

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ============================================================
# FONCTIONS
# ============================================================

def compute_morphology_scores(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    compactness = (
        4 * np.pi * area
    ) / (perimeter ** 2 + 1e-6)

    H, W = mask.shape

    left = mask[:, :W // 2].sum()
    right = mask[:, W // 2:].sum()

    top = mask[:H // 2, :].sum()
    bottom = mask[H // 2:, :].sum()

    asym_lr = abs(left - right) / (left + right + 1e-6)
    asym_tb = abs(top - bottom) / (top + bottom + 1e-6)

    asymmetry = max(asym_lr, asym_tb)

    (_, _), radius = cv2.minEnclosingCircle(cnt)
    diameter = radius * 2

    return {
        "compactness": float(compactness),
        "asymmetry": float(asymmetry),
        "diameter": float(diameter)
    }, cnt


def compute_global_suspicion_score(scores):

    comp = scores["compactness"]
    asym = scores["asymmetry"]
    diam = scores["diameter"]

    comp_score = (1 - comp) * 100
    asym_score = asym * 100
    diam_score = min(diam / 150, 1) * 100

    final_score = (
        0.40 * comp_score +
        0.40 * asym_score +
        0.20 * diam_score
    )

    return min(100, max(0, final_score))


def visualize_border_irregularity(mask, cnt):

    hull = cv2.convexHull(cnt)

    fig, ax = plt.subplots(figsize=(5, 5))

    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)

    cv2.drawContours(vis, [cnt], -1, (255, 0, 0), 2)
    cv2.drawContours(vis, [hull], -1, (0, 255, 0), 2)

    ax.imshow(vis)
    ax.set_title("Rouge = contour réel | Vert = contour convexe")
    ax.axis("off")

    return fig


def visualize_asymmetry_heatmap(mask):

    H, W = mask.shape
    mask_norm = mask.astype(float)

    left = mask_norm[:, :W // 2]
    right = mask_norm[:, W // 2:]
    right_flipped = np.fliplr(right)

    asym_lr = np.abs(left - right_flipped)
    asym_lr = cv2.resize(asym_lr, (W, H))

    top = mask_norm[:H // 2, :]
    bottom = mask_norm[H // 2:, :]
    bottom_flipped = np.flipud(bottom)

    asym_tb = np.abs(top - bottom_flipped)
    asym_tb = cv2.resize(asym_tb, (W, H))

    heatmap = asym_lr + asym_tb

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(heatmap, cmap="hot")
    ax.set_title("Heatmap d'asymétrie")
    ax.axis("off")

    return fig


# ============================================================
# INTERFACE
# ============================================================

st.title("Analyse morphologique de lésions cutanées")

st.sidebar.header("Paramètres")

conf_threshold = st.sidebar.slider(
    "Seuil de confiance",
    0.0,
    1.0,
    0.25,
    0.01
)

imgsz = st.sidebar.selectbox(
    "Taille image",
    [256, 320, 640, 1024],
    index=2
)

st.sidebar.write("Task :", model.task)
st.sidebar.write("Classes :", model.names)

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# ANALYSE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    st.image(
        img_rgb,
        caption="Image originale",
        use_container_width=True
    )

    if st.button("Lancer l'analyse"):

        with st.spinner("Inférence en cours..."):

            img_bgr = cv2.cvtColor(
                img_rgb,
                cv2.COLOR_RGB2BGR
            )

            results = model.predict(
                source=img_bgr,
                conf=conf_threshold,
                imgsz=imgsz,
                verbose=False
            )

            res = results[0]

            st.write(
                "Nombre de détections :",
                len(res.boxes)
            )

            result_img = res.plot()
            result_img = cv2.cvtColor(
                result_img,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                result_img,
                caption="Résultat YOLO",
                use_container_width=True
            )

            # ====================================================
            # MASQUES
            # ====================================================

            if res.masks is None:

                st.warning(
                    "Aucun masque détecté."
                )

            else:

                masks = res.masks.data.cpu().numpy()
                scores_yolo = res.boxes.conf.cpu().numpy()

                best_index = np.argmax(scores_yolo)
                best_mask = masks[best_index]

                binary_mask = (
                    best_mask >= 0.20
                ).astype(np.uint8)

                kernel = np.ones((3, 3), np.uint8)

                binary_mask = cv2.morphologyEx(
                    binary_mask,
                    cv2.MORPH_CLOSE,
                    kernel
                )

                binary_mask = cv2.morphologyEx(
                    binary_mask,
                    cv2.MORPH_OPEN,
                    kernel
                )

                st.image(
                    binary_mask * 255,
                    caption="Masque final",
                    use_container_width=True
                )

                overlay_mask = cv2.resize(
                    binary_mask,
                    (
                        img_rgb.shape[1],
                        img_rgb.shape[0]
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

                mask_color = np.zeros_like(img_rgb)
                mask_color[..., 0] = overlay_mask * 255

                overlay = cv2.addWeighted(
                    img_rgb,
                    1.0,
                    mask_color,
                    0.4,
                    0
                )

                st.image(
                    overlay,
                    caption="Overlay",
                    use_container_width=True
                )

                # ====================================================
                # ANALYSE MORPHOLOGIQUE
                # ====================================================

                scores, cnt = compute_morphology_scores(
                    binary_mask
                )

                if scores is not None:

                    comp = scores["compactness"]
                    asym = scores["asymmetry"]
                    diam = scores["diameter"]

                    suspicion_score = (
                        compute_global_suspicion_score(
                            scores
                        )
                    )

                    st.header(
                        "Analyse morphologique"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Compacité",
                            f"{comp:.3f}"
                        )

                    with col2:
                        st.metric(
                            "Asymétrie",
                            f"{asym:.3f}"
                        )

                    with col3:
                        st.metric(
                            "Diamètre",
                            f"{diam:.1f} px"
                        )

                    st.metric(
                        "Score global de suspicion",
                        f"{suspicion_score:.1f}/100"
                    )

                    st.progress(
                        suspicion_score / 100
                    )

                    if diam < 80 and comp > 0.70:
                        niveau = "faible"
                    else:
                        if suspicion_score >= 70:
                            niveau = "élevé"
                        elif suspicion_score >= 40:
                            niveau = "modéré"
                        else:
                            niveau = "faible"

                    if niveau == "élevé":
                        st.error(
                            "Niveau de complexité morphologique : ÉLEVÉ"
                        )

                    elif niveau == "modéré":
                        st.warning(
                            "Niveau de complexité morphologique : MODÉRÉ"
                        )

                    else:
                        st.success(
                            "Niveau de complexité morphologique : FAIBLE"
                        )

                    st.subheader(
                        "Analyse des bords"
                    )

                    fig1 = visualize_border_irregularity(
                        binary_mask,
                        cnt
                    )

                    st.pyplot(fig1)

                    st.subheader(
                        "Heatmap d'asymétrie"
                    )

                    fig2 = visualize_asymmetry_heatmap(
                        binary_mask
                    )

                    st.pyplot(fig2)

                    st.info(
                        """
                        Ce score reflète uniquement la
                        complexité morphologique de la
                        lésion (forme, asymétrie et taille).

                        Il ne constitue pas un diagnostic
                        médical et ne remplace pas l'avis
                        d'un dermatologue.
                        """
                    )