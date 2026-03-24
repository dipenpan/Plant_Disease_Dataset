import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
    body {
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }

    .stApp {
        background: radial-gradient(circle at top left, #10172b 0%, #0a1020 45%, #060b16 100%);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(17,24,39,0.96), rgba(15,23,42,0.96));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 32px 30px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        margin-bottom: 1.4rem;
    }

    .soft-card {
        background: rgba(17,24,39,0.92);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(30,41,59,0.92));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        min-height: 120px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #cbd5e1;
    }

    .result-good {
        background: linear-gradient(135deg, rgba(6,95,70,0.35), rgba(17,24,39,0.96));
        border: 1px solid rgba(16,185,129,0.35);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
        margin-top: 0.75rem;
    }

    .result-bad {
        background: linear-gradient(135deg, rgba(127,29,29,0.28), rgba(17,24,39,0.96));
        border: 1px solid rgba(248,113,113,0.30);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
        margin-top: 0.75rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
        color: #ffffff;
    }

    .muted {
        color: #cbd5e1;
        font-size: 0.97rem;
    }

    .pill {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.25);
        color: #d1fae5;
        font-size: 0.88rem;
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
    }

    .small-space {
        height: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.h5")

model = load_model()

# ---------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def humanize_label(label: str) -> str:
    return label.replace("___", " → ").replace("_", " ").strip()

def split_label(label: str):
    parts = label.split("___")
    if len(parts) == 2:
        plant, condition = parts
    else:
        plant, condition = "Unknown Plant", label
    plant = plant.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    return plant, condition

def get_health_status(label: str) -> str:
    return "Healthy" if "healthy" in label.lower() else "Diseased"

def get_confidence_level(score: float):
    if score >= 80:
        return "High", "✅"
    elif score >= 60:
        return "Moderate", "⚠️"
    return "Low", "❗"

def get_advice(label: str) -> str:
    lower_label = label.lower()

    if "healthy" in lower_label:
        return "The plant looks healthy. Continue regular monitoring, proper watering, and balanced nutrient care."
    if "blight" in lower_label:
        return "Remove affected leaves, avoid overhead watering, and consider an appropriate fungicide if the infection spreads."
    if "rust" in lower_label:
        return "Prune infected sections, improve airflow, and apply a suitable rust control treatment."
    if "mildew" in lower_label:
        return "Reduce humidity, increase sunlight or airflow, and apply mildew treatment if necessary."
    if "bacterial" in lower_label:
        return "Remove infected foliage, sanitize tools, avoid splashing water, and isolate the plant if possible."
    if "leaf mold" in lower_label or "leaf spot" in lower_label:
        return "Reduce leaf wetness, improve ventilation, and remove visibly infected leaves."
    if "virus" in lower_label:
        return "Isolate infected plants, manage insect vectors, and remove badly affected leaves or plants."
    if "mite" in lower_label:
        return "Inspect the underside of leaves, isolate the plant, and use insecticidal soap or miticide if required."

    return "Monitor the plant closely and consult a plant disease guide or agricultural expert for the next treatment step."

def get_disease_explanation(label: str) -> str:
    lower = label.lower()

    if "healthy" in lower:
        return "The leaf does not show strong visual signs of disease."
    if "blight" in lower:
        return "Blight usually appears as brown or dark patches that spread across the leaf."
    if "rust" in lower:
        return "Rust often appears as orange, yellow, or brown pustules or spots on the leaf surface."
    if "mildew" in lower:
        return "Powdery mildew often looks like a white or dusty coating on leaves."
    if "bacterial" in lower:
        return "Bacterial infections often create water-soaked or dark lesions with irregular edges."
    if "leaf mold" in lower or "leaf spot" in lower:
        return "Leaf spot or mold commonly causes circular or irregular lesions and discoloration."
    if "virus" in lower:
        return "Viral diseases often cause curling, mosaic patterns, yellowing, or stunted growth."
    if "mite" in lower:
        return "Mite damage may appear as tiny speckles, bronzing, or patchy leaf discoloration."

    return "This condition affects the appearance and health of the leaf. Verify with another image if needed."

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    original = image.copy()
    resized = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(resized)
    input_arr = np.expand_dims(input_arr, axis=0)
    return original, input_arr

def predict_image(uploaded_file):
    original_image, input_arr = preprocess_image(uploaded_file)
    preds = model.predict(input_arr, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:3]

    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "label": CLASS_NAMES[int(idx)],
            "score": float(preds[int(idx)]) * 100
        })

    return original_image, results

def image_to_bytes(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def save_history(image_name, top_result):
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    st.session_state.prediction_history.insert(0, {
        "image_name": image_name,
        "label": top_result["label"],
        "score": top_result["score"]
    })

    st.session_state.prediction_history = st.session_state.prediction_history[:5]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.markdown("## 🌿 Dashboard")
st.sidebar.markdown("Use this AI app to detect plant leaf diseases from an uploaded image.")

app_mode = st.sidebar.radio(
    "Select Page",
    ["Home", "About", "Disease Recognition"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Highlights")
st.sidebar.markdown("""
<div class="pill">38 classes</div>
<div class="pill">TensorFlow</div>
<div class="pill">Streamlit UI</div>
<div class="pill">Image-based AI</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
if app_mode == "Home":
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.45rem;">🌿 Plant Disease Recognition System</h1>
        <p class="muted">
            A polished AI-powered web application for detecting plant leaf diseases from images.
            Built for real-time analysis, clear results, and a cleaner user experience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        st.image("home.jpg", use_column_width=True)
    except Exception:
        st.info("Add 'home.jpg' to show the homepage banner image.")

    st.markdown('<div class="small-space"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">38</div>
            <div class="metric-label">Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">AI</div>
            <div class="metric-label">Deep Learning Model</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Live</div>
            <div class="metric-label">Image Analysis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="small-space"></div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="soft-card">
            <div class="section-title">🚀 How It Works</div>
            <div class="muted">
                1. Upload a plant leaf image.<br>
                2. The model analyzes visual disease patterns.<br>
                3. The app predicts the most likely class.<br>
                4. You get confidence scores and treatment guidance.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="soft-card">
            <div class="section-title">✅ Why This Version Feels Better</div>
            <div class="muted">
                • Cleaner layout and stronger visual hierarchy<br>
                • Top 3 prediction confidence view<br>
                • Clear health status and treatment hints<br>
                • Better suited for portfolio and deployment
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------
elif app_mode == "About":
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.45rem;">📘 About This Project</h1>
        <p class="muted">
            This application classifies plant leaf diseases using a trained deep learning model.
            It is designed as a practical AI showcase for plant health monitoring.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
        <div class="soft-card">
            <div class="section-title">📊 Dataset Summary</div>
            <div class="muted">
                • Train: 70,295 images<br>
                • Validation: 17,572 images<br>
                • Test: 33 images
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="soft-card">
            <div class="section-title">🌱 Plant Categories</div>
            <div class="muted">
                Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper,
                Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="soft-card">
            <div class="section-title">🛠 Tech Stack</div>
            <div class="muted">
                Python, TensorFlow, Streamlit, NumPy, PIL
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="soft-card">
            <div class="section-title">🎯 Project Goal</div>
            <div class="muted">
                To support early disease identification through image-based AI and present
                the result in a user-friendly, deployment-ready interface.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# DISEASE RECOGNITION
# ---------------------------------------------------
elif app_mode == "Disease Recognition":
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.45rem;">🩺 Disease Recognition</h1>
        <p class="muted">
            Upload a plant leaf image and run AI-powered disease detection.
            This version provides a cleaner diagnosis panel, confidence level, and top predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("⚡ AI-powered plant disease detection with real-time analysis")

    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.warning("Please upload an image to begin analysis.")
        st.caption("💡 Tip: Use a clear, well-lit image of a single leaf for best results.")

    else:
        preview_col, result_col = st.columns([1, 1.2])

        with preview_col:
            st.markdown("""
            <div class="soft-card">
                <div class="section-title">🖼 Uploaded Image</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)

            analyze = st.button("Run AI Diagnosis")

        with result_col:
            st.markdown("""
            <div class="soft-card">
                <div class="section-title">🔍 AI Prediction Panel</div>
                <div class="muted">
                    Click <strong>Run AI Diagnosis</strong> to generate diagnosis results,
                    confidence level, top predictions, and guidance.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if analyze:
                with st.spinner("Analyzing image..."):
                    original_image, results = predict_image(uploaded_file)

                top_result = results[0]
                top_label = top_result["label"]
                top_score = top_result["score"]

                second_score = results[1]["score"] if len(results) > 1 else 0
                score_gap = top_score - second_score

                plant_name, condition_name = split_label(top_label)
                advice = get_advice(top_label)
                explanation = get_disease_explanation(top_label)
                health_status = get_health_status(top_label)
                confidence_level, confidence_icon = get_confidence_level(top_score)

                save_history(uploaded_file.name, top_result)

                box_class = "result-good" if health_status == "Healthy" else "result-bad"
                icon = "🌱" if health_status == "Healthy" else "⚠️"

                st.markdown(f"""
                <div class="{box_class}">
                    <h3>{icon} AI Diagnosis</h3>
                    <p><strong>Plant:</strong> {plant_name}</p>
                    <p><strong>Condition:</strong> {condition_name}</p>
                    <p><strong>Explanation:</strong> {explanation}</p>
                    <p><strong>Confidence Level:</strong> {confidence_level}</p>
                    <p><strong>Suggestion:</strong> {advice}</p>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)

                with m1:
                    if health_status == "Healthy":
                        st.success("🌱 Healthy")
                    else:
                        st.error("⚠️ Diseased")

                with m2:
                    if confidence_level == "High":
                        st.success(f"{confidence_icon} {confidence_level}")
                    elif confidence_level == "Moderate":
                        st.warning(f"{confidence_icon} {confidence_level}")
                    else:
                        st.error(f"{confidence_icon} {confidence_level}")

                with m3:
                    st.info(f"📊 Confidence Score: {top_score:.2f}%")

                if confidence_level == "Low":
                    st.warning("Low confidence prediction. Try uploading a clearer image of a single leaf.")
                elif confidence_level == "Moderate":
                    st.warning("Moderate confidence prediction. A second image may improve reliability.")

                if score_gap < 10:
                    st.warning("Top predictions are very close. The model is less certain. Try another image.")

                st.markdown("### 🏆 Top 3 Predictions")

                for i, item in enumerate(results, start=1):
                    label = humanize_label(item["label"])
                    score = item["score"]

                    if i == 1:
                        st.success(f"🥇 {label}")
                    elif i == 2:
                        st.write(f"**2. {label}**")
                    else:
                        st.write(f"3. {label}")

                    st.progress(min(int(score), 100))
                    st.caption(f"{score:.2f}% confidence")

                st.download_button(
                    label="Download Uploaded Image",
                    data=image_to_bytes(original_image),
                    file_name="uploaded_leaf.png",
                    mime="image/png"
                )

                with st.expander("See raw predicted labels"):
                    for item in results:
                        st.write(f"{item['label']} — {item['score']:.2f}%")

    if "prediction_history" in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("Recent Predictions")
        for item in st.session_state.prediction_history:
            st.write(
                f"• **{item['image_name']}** → {humanize_label(item['label'])} "
                f"({item['score']:.2f}%)"
            )
