# Handwritten Digit Recognizer (MNIST CNN)

A Streamlit app that recognizes hand-drawn digits (0–9). Draw a digit
on the canvas and get an instant prediction with a confidence score.

This version uses a small CNN trained on the **real MNIST dataset**
(60,000 handwritten digit images), not the tiny 8x8 toy dataset — so it
generalizes much better to freehand canvas drawings. Preprocessing
(cropping to the digit's bounding box, padding, and re-centering by
center of mass) mirrors how the original MNIST images were prepared,
which is what makes canvas predictions accurate.

Test accuracy: ~99%.

## Files (needed for deployment)
- `app.py` — the Streamlit app
- `model.keras` — pre-trained CNN (already trained, ready to use)
- `requirements.txt` — dependencies

## Files (only needed if you want to retrain)
- `train_model.py` — retrains the CNN from scratch
- `mnist_loader.py` — loads the raw MNIST files
- `mnist_raw/` — the raw MNIST dataset (~12MB)

You do **not** need to retrain to deploy — `model.keras` is already
built and tested.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a new GitHub repo and push at minimum: `app.py`,
   `model.keras`, `requirements.txt` (include the retraining files too
   if you want them in the repo — they're not required at runtime).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick the repo/branch, set the main file to
   `app.py`.
4. Click **Deploy**. You'll get a public URL like
   `https://your-app-name.streamlit.app` — use that as your demo link.

## Tips for good predictions
- Draw the digit fairly large and centered in the box
- Use a thick stroke (the canvas is already set to 18px)
- One digit at a time
