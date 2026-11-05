#!/usr/bin/env python3

import os
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(np.float32)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    return -np.mean(
        np.sum(y_true * np.log(y_pred), axis=1)
    )


# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

print("Loading MNIST...")

X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False
)

X = X.astype(np.float32) / 255.0
y = y.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=10000,
    random_state=42
)

Y_train = one_hot(y_train)

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# -------------------------------------------------------
# Network Architecture
# -------------------------------------------------------

INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

# He initialization

W1 = (
    np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
    * np.sqrt(2.0 / INPUT_SIZE)
)

b1 = np.zeros((1, HIDDEN_SIZE))

W2 = (
    np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)
    * np.sqrt(2.0 / HIDDEN_SIZE)
)

b2 = np.zeros((1, OUTPUT_SIZE))


# -------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------

LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCHS = 20

n_samples = X_train.shape[0]


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------

for epoch in range(EPOCHS):

    permutation = np.random.permutation(n_samples)

    X_train = X_train[permutation]
    Y_train = Y_train[permutation]

    epoch_loss = 0.0
    batch_count = 0

    for start in range(0, n_samples, BATCH_SIZE):

        end = start + BATCH_SIZE

        X_batch = X_train[start:end]
        Y_batch = Y_train[start:end]

        m = X_batch.shape[0]

        # -------------------------
        # Forward Pass
        # -------------------------

        Z1 = X_batch @ W1 + b1
        A1 = relu(Z1)

        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        loss = cross_entropy(Y_batch, A2)

        epoch_loss += loss
        batch_count += 1

        # -------------------------
        # Backpropagation
        # -------------------------

        dZ2 = (A2 - Y_batch) / m

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)

        dW1 = X_batch.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # -------------------------
        # Gradient Descent
        # -------------------------

        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2

        W1 -= LEARNING_RATE * dW1
        b1 -= LEARNING_RATE * db1

    # -------------------------
    # Evaluate on Test Set
    # -------------------------

    Z1_test = X_test @ W1 + b1
    A1_test = relu(Z1_test)

    Z2_test = A1_test @ W2 + b2
    A2_test = softmax(Z2_test)

    predictions = np.argmax(A2_test, axis=1)

    accuracy = np.mean(predictions == y_test)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {epoch_loss / batch_count:.4f} | "
        f"Accuracy: {accuracy:.4%}"
    )


# -------------------------------------------------------
# Save Model
# -------------------------------------------------------

os.makedirs("models", exist_ok=True)

np.savez(
    "models/weights.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print()
print("Training complete.")
print("Saved model to: models/weights.npz")

# -------------------------------------------------------
# Final Test Accuracy
# -------------------------------------------------------

final_accuracy = np.mean(predictions == y_test)

print(f"Final accuracy: {final_accuracy:.4%}")
