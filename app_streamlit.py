import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import json

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Hand Gesture AI",
    page_icon="✋",
    layout="centered"
)

# ================================
# CUSTOM CSS (UI MAGIC ✨)
# ================================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    color: white;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}

.pred-box {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.4rem;
    color: white;
    font-weight: bold;
}

.conf-box {
    background: #1e293b;
    padding: 10px;
    border-radius: 10px;
    color: #38bdf8;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODEL
# ================================
MODEL_PATH = "best_gesture_model_mobile.keras"
LABELS_PATH = "labels.json"

@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

# ================================
# LOAD LABELS (FIXED)
# ================================
with open(LABELS_PATH) as f:
    class_indices = json.load(f)

sorted_labels = sorted(class_indices.items(), key=lambda x: x[1])

emoji_map = {
    "palm": "✋", "l": "🤟", "fist": "✊", "fist_moved": "✊",
    "thumb": "👍", "index": "☝", "ok": "👌",
    "palm_moved": "✋", "c": "🤏", "down": "👇"
}

labels = [
    f"{key.split('_',1)[1].capitalize()} {emoji_map.get(key.split('_',1)[1], '')}"
    for key, _ in sorted_labels
]

# ================================
# MEDIAPIPE
# ================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ================================
# UI HEADER
# ================================
st.markdown('<div class="title">✋ Hand Gesture Recognition</div>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("📤 Upload Gesture Image", type=["jpg", "png", "jpeg"])

# ================================
# PROCESS IMAGE
# ================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    results = hands.process(image_np)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = image_np.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin, xmax = int(min(x_coords)*w), int(max(x_coords)*w)
            ymin, ymax = int(min(y_coords)*h), int(max(y_coords)*h)

            pad = 20
            xmin, ymin = max(0, xmin-pad), max(0, ymin-pad)
            xmax, ymax = min(w, xmax+pad), min(h, ymax+pad)

            hand_crop = image_np[ymin:ymax, xmin:xmax]
            hand_crop = cv2.resize(hand_crop, (128,128))

            with col2:
                st.image(hand_crop, caption="Detected Hand", use_container_width=True)

            # Predict
            img_input = preprocess_input(hand_crop)
            img_input = np.expand_dims(img_input, axis=0)

            preds = model.predict(img_input, verbose=0)
            class_id = np.argmax(preds)
            confidence = np.max(preds)

            label = labels[class_id]

            st.markdown("---")

            # ================================
            # RESULT UI
            # ================================
            st.markdown(f'<div class="pred-box">Prediction: {label}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="conf-box">Confidence: {confidence*100:.2f}%</div>', unsafe_allow_html=True)

            st.progress(float(confidence))

            with st.expander("📊 See Probability Breakdown"):
                chart_data = {labels[i]: float(preds[0][i]) for i in range(len(labels))}
                st.bar_chart(chart_data)

    else:
        st.error("❌ No hand detected. Try a clearer image.")

else:
    st.info("Upload an image to start")

