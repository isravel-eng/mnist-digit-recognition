"""Streamlit app for predicting a handwritten digit with a trained MNIST CNN."""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "digit_cnn.keras"

st.set_page_config(page_title="MNIST Digit Recognition", page_icon="🔢", layout="centered")
st.title("MNIST Digit Recognition")
st.write(
    "Upload one handwritten digit (0–9). The app converts it to a centered, "
    "28 × 28 grayscale image before running the CNN."
)


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    """Load the trained model once per Streamlit session."""
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(uploaded_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Convert an uploaded image to the MNIST-like (1, 28, 28, 1) input shape.

    The preprocessing assumes one digit with a reasonably plain background. It
    converts to grayscale, normalizes background polarity, crops the visible
    digit, preserves its aspect ratio, and centers it on a black 28×28 canvas.
    """
    image = ImageOps.exif_transpose(uploaded_image)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    pixels = np.asarray(gray, dtype=np.uint8)

    # MNIST has a dark background and a bright digit. Invert typical white-paper
    # images by inspecting the image border rather than the digit's overall mean.
    border = np.concatenate(
        [pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]]
    )
    if float(border.mean()) > 127:
        pixels = 255 - pixels

    # Crop empty margins so a digit photographed or drawn on a large canvas does
    # not become too small when resized.
    foreground_yx = np.argwhere(pixels > 20)
    canvas = Image.new("L", (28, 28), color=0)
    if foreground_yx.size:
        y_min, x_min = foreground_yx.min(axis=0)
        y_max, x_max = foreground_yx.max(axis=0) + 1
        digit = Image.fromarray(pixels).crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        width, height = digit.size
        scale = min(20 / max(width, 1), 20 / max(height, 1))
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        digit = digit.resize(new_size, Image.Resampling.LANCZOS)
        offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
        canvas.paste(digit, offset)

    normalized = np.asarray(canvas, dtype=np.float32) / 255.0
    model_input = normalized[np.newaxis, ..., np.newaxis]
    return model_input, canvas


uploaded_file = st.file_uploader(
    "Choose a digit image", type=["png", "jpg", "jpeg"],
    help="Use an image containing one digit. Clear, high-contrast digits work best.",
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file)
        model_input, processed_image = preprocess_image(original)
        left, right = st.columns(2)
        with left:
            st.image(original, caption="Uploaded image", use_container_width=True)
        with right:
            st.image(processed_image, caption="Model input (28 × 28)", width=180)

        if not MODEL_PATH.exists():
            st.error(
                "Trained model not found at `models/digit_cnn.keras`. "
                "Run all cells in `notebooks/mnist_training.ipynb` first, "
                "or restore the included model file."
            )
        else:
            model = load_model(str(MODEL_PATH))
            probabilities = model.predict(model_input, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_digit])
            st.subheader(f"Predicted digit: {predicted_digit}")
            st.metric("Model confidence", f"{confidence:.1%}")
            st.caption("Confidence is the model's softmax score, not a guarantee of correctness.")
            st.write("Scores for digits 0–9")
            st.bar_chart(probabilities)
    except Exception as exc:
        st.error(f"Could not process this image or load the model: {exc}")
else:
    st.info("Upload a single handwritten digit to get a prediction.")
