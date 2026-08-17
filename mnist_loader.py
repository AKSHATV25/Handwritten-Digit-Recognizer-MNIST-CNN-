import gzip
import numpy as np


def load_images(path, num_images):
    with gzip.open(path, "rb") as f:
        f.read(16)  # header
        buf = f.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num_images, 28, 28)


def load_labels(path, num_labels):
    with gzip.open(path, "rb") as f:
        f.read(8)  # header
        buf = f.read(num_labels)
        return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(dir_="mnist_raw"):
    X_train = load_images(f"{dir_}/train-images.gz", 60000)
    y_train = load_labels(f"{dir_}/train-labels.gz", 60000)
    X_test = load_images(f"{dir_}/test-images.gz", 10000)
    y_test = load_labels(f"{dir_}/test-labels.gz", 10000)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print("train:", X_train.shape, y_train.shape)
    print("test:", X_test.shape, y_test.shape)
    print("label sample:", y_train[:10])
