#!/usr/bin/env python3

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Activations
# -------------------------------------------------------

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# -------------------------------------------------------
# Load model
# -------------------------------------------------------

weights = np.load("models/weights.npz")

W1 = weights["W1"]
b1 = weights["b1"]

W2 = weights["W2"]
b2 = weights["b2"]


# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------

def predict(image_array):
    z1 = image_array @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    probs = softmax(z2)

    digit = np.argmax(probs)

    return digit, probs[0]


# -------------------------------------------------------
# Load image
# -------------------------------------------------------

if len(sys.argv) != 2:
    print("Usage:")
    print("    python predict.py image.png")
    sys.exit(1)

image_path = sys.argv[1]

img = Image.open(image_path).convert("L")
img = img.resize((28, 28))

image = np.array(img, dtype=np.float32)

# Normalize to [0,1]
image /= 255.0

# Invert colors if needed
# MNIST expects white digit on black background.
# Uncomment if your images are black digits on white paper.
image = 1.0 - image

x = image.reshape(1, 784)

digit, probs = predict(x)

# -------------------------------------------------------
# Output
# -------------------------------------------------------

print()
print(f"Predicted digit: {digit}")
print()

for i, p in enumerate(probs):
    print(f"{i}: {p:.4f}")

print()

# Show image
plt.imshow(image.reshape(28, 28), cmap="gray")
plt.title(f"Prediction: {digit}")
plt.axis("off")
plt.show()