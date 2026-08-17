"""
Trains a small CNN on the real MNIST dataset (28x28 handwritten digits,
60,000 training images) and saves it as model.keras.

This is the dataset digit-recognizer demos are actually built on -- unlike
sklearn's built-in 8x8 toy "digits" dataset, MNIST is large and diverse
enough to generalize to freehand canvas drawings.

Run once locally: python train_model.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from mnist_loader import load_mnist


def main():
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=6,
        batch_size=128,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    model.save("model.keras")
    print("Saved model.keras")


if __name__ == "__main__":
    main()
