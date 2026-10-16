from pathlib import Path
import json

import requests
import streamlit as st

from src.components.header import (
    render_top_header,
    page_header,
    section_header,
    medical_disclaimer,
)

from src.components.sidebar import render_sidebar

from src.components.cards import (
    metric_card,
    feature_card,
)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

METRICS_PATH = MODELS_DIR / "verification_metrics_v3.json"

CONFUSION_MATRIX_PATH = (
    MODELS_DIR / "verification_confusion_matrix_v3.png"
)

MODEL_PATHS = [
    MODELS_DIR / "best_model_v3.keras",
    MODELS_DIR / "final_model_v3.keras",
]

# Flask backend.
API_URL = "http://127.0.0.1:5000"

PREDICTION_ENDPOINT = f"{API_URL}/api/predict"

API_TIMEOUT_SECONDS = 120

st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. SESSION STATE
# ============================================================

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Dark"

if "header_theme_toggle" not in st.session_state:
    st.session_state.header_theme_toggle = (
        st.session_state.ui_theme == "Dark"
    )

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_prediction_key" not in st.session_state:
    st.session_state.last_prediction_key = None


# ============================================================
# 3. THEME COLORS
# ============================================================

THEMES = {
    "Dark": {
        "app_bg": "#081321",
        "surface": "#101f32",
        "surface_hover": "#172b42",
        "text_primary": "#edf5ff",
        "text_secondary": "#b0c1d6",
        "text_muted": "#8296af",
        "border": "rgba(148, 182, 218, 0.17)",
        "border_strong": "rgba(148, 182, 218, 0.32)",
        "input_bg": "#0c1929",
        "sidebar_bg": "#091625",
        "glass_bg": "rgba(14, 29, 48, 0.78)",
        "glass_border": "rgba(148, 200, 240, 0.25)",
        "accent": "#38bdf8",
        "accent_strong": "#0284c7",
        "shadow": "0 12px 32px rgba(0, 0, 0, 0.20)",
        "shadow_hover": "0 18px 38px rgba(0, 0, 0, 0.30)",
    },

    "Light": {
        "app_bg": "#f2f6fc",
        "surface": "#ffffff",
        "surface_hover": "#eef6ff",
        "text_primary": "#172b45",
        "text_secondary": "#50647c",
        "text_muted": "#7789a0",
        "border": "rgba(64, 100, 140, 0.15)",
        "border_strong": "rgba(64, 100, 140, 0.28)",
        "input_bg": "#ffffff",
        "sidebar_bg": "#ffffff",
        "glass_bg": "rgba(255, 255, 255, 0.78)",
        "glass_border": "rgba(255, 255, 255, 0.90)",
        "accent": "#0284c7",
        "accent_strong": "#0369a1",
        "shadow": "0 12px 32px rgba(32, 67, 107, 0.08)",
        "shadow_hover": "0 18px 38px rgba(32, 67, 107, 0.14)",
    },
}


# ============================================================
# 4. THEME CSS
# ============================================================

def load_theme_css():
    """Load CSS and inject the selected theme variables."""

    css_path = ASSETS_DIR / "styles.css"

    if not css_path.is_file():
        st.error(f"Stylesheet not found: {css_path}")
        return

    css_content = css_path.read_text(encoding="utf-8")

    theme = THEMES.get(
        st.session_state.ui_theme,
        THEMES["Dark"],
    )

    theme_variables = "\n".join(
        f"--{name.replace('_', '-')}: {value};"
        for name, value in theme.items()
    )

    theme_css = f"""
    <style>
    {css_content}

    .stApp {{
        {theme_variables}
    }}

    section[data-testid="stSidebar"] {{
        {theme_variables}
    }}

    .stApp,
    section[data-testid="stSidebar"] {{
        color: var(--text-primary);
    }}
    </style>
    """

    st.markdown(
        theme_css,
        unsafe_allow_html=True,
    )


# ============================================================
# 5. MODEL AND METRICS HELPERS
# ============================================================

def model_exists():
    """Check whether at least one expected model file exists."""

    return any(path.is_file() for path in MODEL_PATHS)


def load_metrics():
    """Load saved evaluation metrics with safe fallbacks."""

    fallback = {
        "accuracy": 0.8574,
        "precision": 0.8265,
        "recall": 0.9769,
        "sensitivity": 0.9769,
        "specificity": 0.6581,
        "f1_score": 0.8954,
        "roc_auc": 0.9652,
        "total_images": 624,
        "normal_images": 234,
        "pneumonia_images": 390,
    }

    if not METRICS_PATH.is_file():
        return fallback

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            loaded = json.load(file)

        if not isinstance(loaded, dict):
            return fallback

        result = fallback.copy()
        result.update(loaded)

        aliases = {
            "accuracy": ["accuracy", "test_accuracy"],
            "precision": ["precision", "test_precision"],
            "recall": [
                "recall",
                "sensitivity",
                "test_recall",
                "test_sensitivity",
            ],
            "sensitivity": [
                "sensitivity",
                "recall",
                "test_sensitivity",
                "test_recall",
            ],
            "specificity": [
                "specificity",
                "test_specificity",
            ],
            "f1_score": [
                "f1_score",
                "f1",
                "test_f1",
            ],
            "roc_auc": [
                "roc_auc",
                "auc",
                "test_auc",
            ],
        }

        for target, candidates in aliases.items():
            for key in candidates:
                if key in loaded:
                    try:
                        result[target] = float(loaded[key])
                        break
                    except (TypeError, ValueError):
                        continue

        return result

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return fallback


METRICS = load_metrics()


def safe_float(value, default=0.0):
    """Convert values to floats safely."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percentage(value):
    """Format metric values as percentages."""

    try:
        number = float(value)

        if number <= 1:
            number *= 100

        return f"{number:.2f}%"

    except (TypeError, ValueError):
        return "N/A"


# ============================================================
# 6. FLASK API CONNECTION
# ============================================================

def analyze_with_api(uploaded_file):
    """
    Send the uploaded image to Flask and return its JSON result.

    Flask expects the file in a multipart/form-data field
    named 'image'.
    """

    if uploaded_file is None:
        st.warning("Please upload a chest X-ray image.")
        return None

    try:
        file_bytes = uploaded_file.getvalue()

        if not file_bytes:
            st.error("The uploaded file is empty.")
            return None

        files = {
            "image": (
                uploaded_file.name,
                file_bytes,
                uploaded_file.type or "application/octet-stream",
            )
        }

        response = requests.post(
            PREDICTION_ENDPOINT,
            files=files,
            timeout=API_TIMEOUT_SECONDS,
        )

        try:
            data = response.json()
        except ValueError:
            st.error(
                "The Flask API returned an invalid response."
            )
            return None

        if not response.ok:
            st.error(
                data.get(
                    "error",
                    f"API request failed: HTTP {response.status_code}",
                )
            )
            return None

        if not data.get("success"):
            st.error(
                data.get(
                    "error",
                    "The API could not analyze the image.",
                )
            )
            return None

        prediction = data.get("prediction")

        if not isinstance(prediction, dict):
            st.error(
                "The API response does not contain a valid prediction."
            )
            return None

        return {
            **prediction,
            "filename": data.get(
                "filename",
                uploaded_file.name,
            ),
            "notice": data.get("notice", ""),
        }

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the Flask API. "
            "Open a second terminal and run: python api.py"
        )
        return None

    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out. "
            "Check the Flask terminal and try again."
        )
        return None

    except requests.exceptions.RequestException as error:
        st.error(f"API request failed: {error}")
        return None


# ============================================================
# 7. SHARED COMPONENTS
# ============================================================

def show_metric_row():
    """Display the main evaluation metrics."""

    columns = st.columns(4)

    metric_definitions = [
        (
            "Accuracy",
            METRICS.get("accuracy"),
            "🎯",
            "Overall test-set accuracy",
        ),
        (
            "Sensitivity",
            METRICS.get(
                "sensitivity",
                METRICS.get("recall"),
            ),
            "🫁",
            "Pneumonia cases identified",
        ),
        (
            "Specificity",
            METRICS.get("specificity"),
            "🔍",
            "Normal cases correctly identified",
        ),
        (
            "ROC-AUC",
            METRICS.get("roc_auc"),
            "📈",
            "Classification discrimination",
        ),
    ]

    for column, (title, value, icon, description) in zip(
        columns,
        metric_definitions,
    ):
        with column:
            metric_card(
                title=title,
                value=format_percentage(value),
                icon=icon,
                description=description,
            )


def show_footer():
    """Display the application footer."""

    st.divider()

    st.caption(
        "MedVision AI · Educational Research Prototype · "
        "Not for clinical diagnosis"
    )


# ============================================================
# 8. DASHBOARD
# ============================================================

def show_dashboard():
    page_header(
        "Medical Imaging Dashboard",
        (
            "Explore your AI-powered chest X-ray research workspace, "
            "review model evaluation results, and analyze an image."
        ),
        badge="RESEARCH PROTOTYPE",
    )

    with st.container(border=True):
        st.markdown("### 🩺 Welcome to MedVision AI")

        st.write(
            "An experimental chest X-ray classification application "
            "powered by your trained DenseNet121 model. Upload an "
            "image to inspect the model's predicted class and output "
            "scores."
        )

        st.caption(
            "This tool is intended for educational research and "
            "does not provide a medical diagnosis."
        )

    st.write("")

    section_header(
        "Model Evaluation Overview",
        "Previously recorded results from the test dataset.",
    )

    show_metric_row()

    st.write("")

    section_header(
        "Explore Your Workspace",
        "Choose a section from the sidebar to get started.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        feature_card(
            "Analyze an X-ray",
            "Upload a chest X-ray and inspect the model output.",
            icon="🩻",
        )

    with col2:
        feature_card(
            "Model Performance",
            "Review saved evaluation metrics and the confusion matrix.",
            icon="📊",
        )

    with col3:
        feature_card(
            "About the Project",
            "Explore the model architecture and project limitations.",
            icon="🧠",
        )

    st.write("")

    section_header(
        "Recent Analyses",
        "The latest predictions from this application session.",
    )

    history = st.session_state.prediction_history

    if history:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No images have been analyzed in this session. "
            "Open Analyze X-ray to begin."
        )

    medical_disclaimer()


# ============================================================
# 9. X-RAY ANALYSIS
# ============================================================

def show_analyze_page():
    page_header(
        "Analyze Chest X-ray",
        (
            "Upload an image and send it to the Flask API, "
            "which runs the existing DenseNet121 prediction pipeline."
        ),
        badge="AI ANALYSIS",
    )

    st.info(
        "This research prototype cannot confirm or exclude pneumonia "
        "in a real patient."
    )

    section_header(
        "Upload Your Image",
        "Supported formats: JPG, JPEG, and PNG.",
    )

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Select a chest X-ray image from your device.",
        key="xray_upload",
    )

    if uploaded_file is None:
        with st.container(border=True):
            st.markdown("### 📤 Ready for image analysis")

            st.write(
                "Select an image above. Its preview and analysis "
                "controls will appear when the upload is complete."
            )

        st.session_state.last_prediction = None
        st.session_state.last_prediction_key = None

        return

    # Identify the currently selected upload.
    image_key = (
        uploaded_file.name,
        len(uploaded_file.getvalue()),
    )

    # Clear old results when a different file is selected.
    if st.session_state.last_prediction_key != image_key:
        st.session_state.last_prediction = None
        st.session_state.last_prediction_key = None

    preview_column, details_column = st.columns(
        [1.1, 1],
        gap="large",
    )

    with preview_column:
        section_header("Image Preview")

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with details_column:
        section_header("Image Information")

        with st.container(border=True):
            st.write(f"**Filename:** {uploaded_file.name}")

            st.write(
                f"**File size:** {uploaded_file.size / 1024:.1f} KB"
            )

            st.write("**Format:** Chest X-ray image")

            st.write("**Model:** DenseNet121")

            st.write("**Backend:** Flask REST API")

            st.caption(
                "The Flask API uses the same DenseNet121 "
                "preprocessing as your existing predictor."
            )

        analyze_button = st.button(
            "🔬 Analyze X-ray",
            type="primary",
            use_container_width=True,
            key="analyze_xray_button",
        )

    if analyze_button:
        with st.spinner(
            "Sending image to MedVision AI for analysis..."
        ):
            result = analyze_with_api(uploaded_file)

        if result is not None:
            required_keys = {
                "predicted_class",
                "normal_score",
                "pneumonia_score",
            }

            missing = required_keys - set(result)

            if missing:
                st.error(
                    "The API response is missing: "
                    + ", ".join(sorted(missing))
                )
                return

            result["filename"] = uploaded_file.name

            st.session_state.last_prediction = result
            st.session_state.last_prediction_key = image_key

            history_entry = {
                "Filename": uploaded_file.name,
                "Predicted Class": str(
                    result["predicted_class"]
                ),
                "NORMAL Score": (
                    f"{safe_float(result['normal_score']):.4f}"
                ),
                "PNEUMONIA Score": (
                    f"{safe_float(result['pneumonia_score']):.4f}"
                ),
            }

            st.session_state.prediction_history.insert(
                0,
                history_entry,
            )

            st.session_state.prediction_history = (
                st.session_state.prediction_history[:10]
            )

    # Show saved result after Streamlit reruns.
    result_to_display = st.session_state.last_prediction

    if (
        result_to_display is not None
        and st.session_state.last_prediction_key == image_key
    ):
        display_prediction_result(result_to_display)


# ============================================================
# 10. DISPLAY PREDICTION
# ============================================================

def display_prediction_result(result):
    """Display the JSON result returned by the Flask API."""

    predicted_class = str(
        result.get("predicted_class", "Unknown")
    ).strip().upper()

    normal_score = safe_float(
        result.get("normal_score")
    )

    pneumonia_score = safe_float(
        result.get("pneumonia_score")
    )

    confidence = safe_float(
        result.get(
            "confidence",
            max(normal_score, pneumonia_score) * 100,
        )
    )

    st.write("")

    section_header(
        "Analysis Result",
        "Results returned by the Flask prediction API.",
    )

    with st.container(border=True):
        if predicted_class == "NORMAL":
            st.success("Model classification: NORMAL")

        elif predicted_class == "PNEUMONIA":
            st.warning("Model classification: PNEUMONIA")

        else:
            st.info(
                f"Model classification: {predicted_class}"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card(
                "Predicted Class",
                predicted_class,
                icon="🩺",
                description="Class selected by the model.",
            )

        with col2:
            metric_card(
                "Model Confidence",
                f"{confidence:.2f}%",
                icon="🎯",
                description=(
                    "Maximum of the two model output scores."
                ),
            )

        with col3:
            metric_card(
                "Model",
                "DenseNet121",
                icon="🧠",
                description="Trained binary classifier.",
            )

        st.write("")

        section_header(
            "Class Scores",
            "Model output scores for each class.",
        )

        score_col1, score_col2 = st.columns(2)

        with score_col1:
            st.markdown("#### 🟢 NORMAL")

            st.progress(
                max(
                    0.0,
                    min(1.0, normal_score),
                )
            )

            st.write(
                f"Output score: {normal_score:.4f}"
            )

        with score_col2:
            st.markdown("#### 🟠 PNEUMONIA")

            st.progress(
                max(
                    0.0,
                    min(1.0, pneumonia_score),
                )
            )

            st.write(
                f"Output score: {pneumonia_score:.4f}"
            )

        st.caption(
            "The model outputs are not necessarily calibrated "
            "probabilities and should not be interpreted as a "
            "medical diagnosis."
        )

    report = (
        "MedVision AI — Experimental Analysis Report\n"
        "============================================\n"
        f"Filename: {result.get('filename', 'Unknown')}\n"
        f"Predicted class: {predicted_class}\n"
        f"Model confidence: {confidence:.2f}%\n"
        f"NORMAL output score: {normal_score:.6f}\n"
        f"PNEUMONIA output score: {pneumonia_score:.6f}\n\n"
        "Backend: Flask REST API\n"
        "Model: DenseNet121\n\n"
        "Disclaimer: Educational research prototype only. "
        "This result is not a medical diagnosis.\n"
    )

    st.download_button(
        "⬇️ Download Analysis Report",
        data=report,
        file_name="medvision_analysis_report.txt",
        mime="text/plain",
    )

    medical_disclaimer()


# ============================================================
# 11. MODEL PERFORMANCE
# ============================================================

def show_model_performance():
    page_header(
        "Model Performance",
        (
            "Explore the saved evaluation results for the "
            "DenseNet121 chest X-ray classifier."
        ),
        badge="EVALUATION",
    )

    st.info(
        "These are previously recorded evaluation results. "
        "Opening this page does not run a new evaluation."
    )

    section_header(
        "Classification Metrics",
        "Recorded performance on the test dataset.",
    )

    show_metric_row()

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Precision",
            format_percentage(METRICS.get("precision")),
            icon="🎯",
            description="Correct positive predictions.",
        )

    with col2:
        metric_card(
            "F1 Score",
            format_percentage(METRICS.get("f1_score")),
            icon="⚖️",
            description="Balance between precision and recall.",
        )

    with col3:
        metric_card(
            "Test Images",
            str(METRICS.get("total_images", 624)),
            icon="🗂️",
            description="Images in the saved evaluation.",
        )

    st.write("")

    section_header(
        "Confusion Matrix",
        "Saved visualization from the evaluation.",
    )

    if CONFUSION_MATRIX_PATH.is_file():
        with st.container(border=True):
            st.image(
                str(CONFUSION_MATRIX_PATH),
                use_container_width=True,
            )
    else:
        st.warning(
            f"Confusion matrix not found: {CONFUSION_MATRIX_PATH}"
        )

    with st.expander("Understand the evaluation metrics"):
        st.markdown(
            """
            - **Accuracy:** Proportion of all test images classified
              correctly.
            - **Precision:** Proportion of positive predictions that
              were correct.
            - **Sensitivity:** Proportion of actual pneumonia cases
              identified.
            - **Specificity:** Proportion of actual normal cases
              identified.
            - **F1 score:** Harmonic mean of precision and recall.
            - **ROC-AUC:** Measures discrimination across
              classification thresholds.
            """
        )

    st.warning(
        "The saved specificity is approximately 65.81%. In that "
        "evaluation, the model incorrectly classified a notable "
        "proportion of normal images as pneumonia. These results "
        "do not establish clinical safety."
    )

    medical_disclaimer()


# ============================================================
# 12. ABOUT PROJECT
# ============================================================

def show_about_page():
    page_header(
        "About MedVision AI",
        (
            "An educational project exploring deep learning for "
            "chest X-ray image classification."
        ),
        badge="PROJECT OVERVIEW",
    )

    section_header("Project Overview")

    with st.container(border=True):
        st.markdown("### 🩺 AI-Powered Medical Image Analysis")

        st.write(
            "MedVision AI is an experimental application built around "
            "a trained DenseNet121 model. It classifies uploaded "
            "chest X-ray images into NORMAL or PNEUMONIA categories."
        )

    section_header("Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        feature_card(
            "Frontend",
            "Streamlit, custom CSS, and responsive layouts.",
            icon="🎨",
        )

    with col2:
        feature_card(
            "Backend",
            "Flask REST API, TensorFlow, Keras, and DenseNet121.",
            icon="🧠",
        )

    with col3:
        feature_card(
            "Image Processing",
            "Pillow, NumPy, and DenseNet121 preprocessing.",
            icon="🩻",
        )

    st.write("")

    section_header("Model Details")

    with st.container(border=True):
        st.write("**Architecture:** DenseNet121")
        st.write("**Input size:** 224 × 224 pixels")
        st.write("**Input channels:** RGB")
        st.write("**Task:** Binary image classification")
        st.write("**Classes:** NORMAL and PNEUMONIA")
        st.write("**Backend:** Flask REST API")

    st.write("")

    section_header("Project File Status")

    files_to_check = [
        ("Best model", MODEL_PATHS[0]),
        ("Final model", MODEL_PATHS[1]),
        ("Evaluation metrics", METRICS_PATH),
        ("Confusion matrix", CONFUSION_MATRIX_PATH),
        ("Stylesheet", ASSETS_DIR / "styles.css"),
    ]

    for label, path in files_to_check:
        if path.is_file():
            st.write(f"✅ **{label}:** Available")
        else:
            st.write(f"⚠️ **{label}:** Not found")

    st.write("")

    section_header("Important Limitations")

    st.markdown(
        """
        - This is an educational research prototype, not a medical
          diagnostic system.
        - Performance may differ on data from other hospitals,
          scanners, or patient populations.
        - Model output scores may not be calibrated probabilities.
        - Predictions must not be used to make treatment decisions.
        """
    )

    medical_disclaimer()


# ============================================================
# 13. MAIN APPLICATION
# ============================================================

def main():
    """Run MedVision AI."""

    load_theme_css()

    # Preserve your existing custom header and theme toggle.
    render_top_header()

    selected_page = render_sidebar(
        model_available=model_exists()
    )

    if selected_page == "Dashboard":
        show_dashboard()

    elif selected_page == "Analyze X-ray":
        show_analyze_page()

    elif selected_page == "Model Performance":
        show_model_performance()

    elif selected_page == "About Project":
        show_about_page()

    show_footer()


if __name__ == "__main__":
    main()