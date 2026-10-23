"""Streamlit app for predicting a handwritten digit with a trained MNIST CNN."""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "digit_cnn.keras"

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
)
st.title("MNIST Digit Recognition")
st.caption(
    "Upload one handwritten digit (0–9). The app prepares a centered 28 × 28 "
    "grayscale image, then predicts the digit using a CNN."
)


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    """Load the trained Keras model once and reuse it across reruns."""
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(uploaded_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Prepare an uploaded image as a centered MNIST-like model input.

    The expected input shape is (1, 28, 28, 1). This routine works best when
    the image contains one clearly visible digit against a plain background.
    """
    image = ImageOps.exif_transpose(uploaded_image)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    pixels = np.asarray(gray, dtype=np.uint8)

    # MNIST uses a dark background and a bright digit. Invert images with a
    # bright border, such as a dark digit written on white paper.
    border = np.concatenate(
        [pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]]
    )
    if float(border.mean()) > 127:
        pixels = 255 - pixels

    # Crop the digit, keep its proportions, and center it on a 28 × 28 canvas.
    foreground_yx = np.argwhere(pixels > 20)
    canvas = Image.new("L", (28, 28), color=0)
    if foreground_yx.size:
        y_min, x_min = foreground_yx.min(axis=0)
        y_max, x_max = foreground_yx.max(axis=0) + 1
        digit = Image.fromarray(pixels).crop(
            (int(x_min), int(y_min), int(x_max), int(y_max))
        )
        width, height = digit.size
        scale = min(20 / max(width, 1), 20 / max(height, 1))
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        digit = digit.resize(new_size, Image.Resampling.LANCZOS)
        offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
        canvas.paste(digit, offset)

    normalized = np.asarray(canvas, dtype=np.float32) / 255.0
    model_input = normalized[np.newaxis, ..., np.newaxis]
    return model_input, canvas


# Keep the upload, original image, and preprocessed image in the left column;
# put the prediction, confidence, and digit-score chart in the right column.
left_panel, right_panel = st.columns([1.15, 1], gap="large")

with left_panel:
    st.subheader("1. Upload and preview")
    uploaded_file = st.file_uploader(
        "Choose a digit image",
        type=["png", "jpg", "jpeg"],
        help=(
            "Choose an image containing one handwritten digit. "
            "Clear, high-contrast digits work best."
        ),
    )

    if uploaded_file is not None:
        try:
            with Image.open(uploaded_file) as image_file:
                uploaded_image = ImageOps.exif_transpose(image_file).copy()

            model_input, processed_image = preprocess_image(uploaded_image)

            original_col, processed_col = st.columns([1.2, 0.8], gap="medium")
            with original_col:
                st.image(
                    uploaded_image,
                    caption="Uploaded image",
                    use_container_width=True,
                )
            with processed_col:
                st.image(
                    processed_image,
                    caption="Model input (28 × 28)",
                    width=168,
                )
            st.caption("The CNN receives the preprocessed 28 × 28 grayscale image.")
        except Exception as exc:
            st.error(f"Could not read or preprocess this image: {exc}")
            st.stop()
    else:
        st.info("Upload a PNG or JPG image containing one handwritten digit.")

with right_panel:
    st.subheader("2. Prediction")

    if uploaded_file is None:
        st.info("Your predicted digit, confidence, and digit scores will appear here.")
    elif not MODEL_PATH.exists():
        st.error(
            "Trained model not found at `models/digit_cnn.keras`. "
            "Run `notebooks/mnist_training.ipynb` or restore the trained model file."
        )
    else:
        try:
            model = load_model(str(MODEL_PATH))
            probabilities = model.predict(model_input, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_digit])

            digit_col, confidence_col = st.columns(2, gap="medium")
            with digit_col:
                st.metric("Predicted digit", str(predicted_digit))
            with confidence_col:
                st.metric("Model confidence", f"{confidence:.1%}")

            st.caption(
                "Confidence is the model's softmax score, not a guarantee of correctness."
            )
            st.write("**Scores for digits 0–9**")
            st.bar_chart(probabilities, height=280)
        except Exception as exc:
            st.error(f"Could not load the model or make a prediction: {exc}")
