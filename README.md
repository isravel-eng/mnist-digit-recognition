# MNIST Digit Recognition

A beginner-friendly handwritten-digit classifier built with TensorFlow/Keras and a Convolutional Neural Network (CNN). The project includes a training notebook, a saved model, evaluation visualizations, and a Streamlit demo.

## Project structure

```text
mnist-digit-recognition/
├── notebooks/
│   └── mnist_training.ipynb
├── models/
│   ├── digit_cnn.keras
│   └── model_info.txt
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── misclassified_examples.png
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Use Python 3.11 where possible. From the repository root, create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

Run from the repository root:

```bash
jupyter notebook notebooks/mnist_training.ipynb
```

Run the notebook cells from top to bottom. It downloads MNIST through Keras, preprocesses the images, trains the CNN for six epochs, evaluates the model on the official test split, and saves/replaces these artifacts:

- `models/digit_cnn.keras`
- `results/confusion_matrix.png`
- `results/training_history.png`
- `results/misclassified_examples.png`

The notebook uses paths based on the repository root, so artifacts stay inside the repository whether the notebook kernel starts in the project root or the `notebooks/` directory.

## Run the demo app

After installing dependencies, and with `models/digit_cnn.keras` present, run:

```bash
streamlit run app.py
```

Upload one handwritten digit as a PNG/JPG/JPEG image. The app converts it to grayscale, adjusts the background polarity, crops the digit, centers it on a 28 × 28 canvas, normalizes pixel values, and passes it to the trained model. Clear, high-contrast images with one digit work best; real-world handwriting may differ from MNIST and can be misclassified.

## Current evaluation artifact

The included confusion matrix shows **9,894 correct predictions out of 10,000 test images (98.94%)**, calculated by summing its diagonal. Re-running training may produce a slightly different score and will overwrite the included model and figures with the new run's outputs.

## Model overview

- Input: `28 × 28 × 1` grayscale image
- Pixel scaling: `float32` values in `[0, 1]`
- Layers: Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D → Flatten → Dense(128) → Dropout(0.3) → Dense(10, softmax)
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Classes: digits `0` through `9`

## Notes

- The `.keras` model file is intentionally included because `app.py` loads it directly. Do not add an ignore rule that hides `models/digit_cnn.keras` from Git.
- The notebook has no saved execution outputs; run it to reproduce the artifacts.
- This is an educational demo, not a guarantee of accurate recognition for arbitrary photos or handwriting styles.
