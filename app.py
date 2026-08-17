import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from scipy.ndimage import center_of_mass, shift
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")


model = load_model()

st.title("✏️ Handwritten Digit Recognizer")
st.write(
    "Draw a single digit (0–9) in the box below, then click **Predict**. "
    "Powered by a CNN trained on the real MNIST dataset (60,000 images, 99%+ test accuracy)."
)

col1, col2 = st.columns([2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("**Instructions**")
    st.markdown(
        "- Draw one digit, roughly centered\n"
        "- Use a thick stroke\n"
        "- Click Predict to see the result\n"
        "- Click the trash icon on the canvas to clear"
    )
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)


def center_via_com(img28):
    """Shift the digit so its center of mass sits at the image center,
    matching how the original MNIST images were prepared."""
    cy, cx = center_of_mass(img28)
    if np.isnan(cy) or np.isnan(cx):
        return img28
    rows, cols = img28.shape
    shift_y = rows / 2.0 - cy
    shift_x = cols / 2.0 - cx
    return shift(img28, (shift_y, shift_x), cval=0)


def preprocess(image_data):
    img = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("L")
    arr = np.array(img)

    ys, xs = np.where(arr > 20)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Pad to square with margin, matching MNIST's ~20px-digit-in-28px-canvas framing
    side = max(cropped.size)
    pad = int(side * 0.4)
    side_padded = side + 2 * pad
    square = Image.new("L", (side_padded, side_padded), color=0)
    square.paste(cropped, (pad + (side - cropped.size[0]) // 2, pad + (side - cropped.size[1]) // 2))

    img28 = square.resize((28, 28), Image.LANCZOS)
    arr28 = np.array(img28).astype("float32")
    arr28 = center_via_com(arr28)

    arr28 = arr28 / 255.0
    return arr28.reshape(1, 28, 28, 1)


if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        x = preprocess(canvas_result.image_data)

        if x is None:
            st.warning("Please draw a digit first.")
        else:
            proba = model.predict(x, verbose=0)[0]
            pred = int(np.argmax(proba))

            st.success(f"### Predicted digit: **{pred}**")
            st.write(f"Confidence: {proba[pred] * 100:.1f}%")

            st.bar_chart({"probability": proba})

st.divider()
st.caption("Model: CNN (TensorFlow/Keras) trained on real MNIST · Built with Streamlit")
