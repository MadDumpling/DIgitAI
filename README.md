# DigitAI

A handwritten digit recognition neural network implemented from scratch using NumPy.

This project trains a simple feedforward neural network on the MNIST dataset and can predict digits from image files.

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- Scikit-Learn
- Pandas
- Pillow

## Installation

Clone the project and enter the directory:

```bash
cd DigitAI
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If pandas is missing:

```bash
pip install pandas
```

---

## Training the Model

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run training:

```bash
python train.py
```

Example output:

```text
Loading MNIST...

Training samples : 60000
Testing samples  : 10000

Epoch 01/20 | Loss: 1.4321 | Accuracy: 82.11%
Epoch 02/20 | Loss: 0.7323 | Accuracy: 90.56%
...
```

When training completes:

```text
Training complete.
Saved model to: models/weights.npz
```

The trained weights are stored in:

```text
models/weights.npz
```

---

## Predicting Digits

Place an image in the project directory or an `images/` folder.

Example:

```text
images/7.png
```

Run:

```bash
python predict.py images/7.png
```

Example output:

```text
Predicted digit: 7

0: 0.0001
1: 0.0000
2: 0.0002
3: 0.0001
4: 0.0000
5: 0.0003
6: 0.0000
7: 0.9987
8: 0.0002
9: 0.0004
```

A preview window will display the processed image and prediction.

---

## Project Structure

```text
DigitAI/
│
├── train.py
├── predict.py
├── requirements.txt
├── setup.sh
│
├── models/
│   └── weights.npz
│
└── images/
    └── example.png
```

---

## Network Architecture

```text
Input Layer   : 784 neurons (28×28 pixels)
Hidden Layer  : 128 neurons (ReLU)
Output Layer  : 10 neurons (Softmax)
```

Training uses:

- ReLU activation
- Softmax output
- Cross-entropy loss
- Backpropagation
- Mini-batch gradient descent

---

## Troubleshooting

### ModuleNotFoundError: No module named 'pandas'

Install pandas:

```bash
pip install pandas
```

### Virtual environment not activated

Activate it:

```bash
source .venv/bin/activate
```

### Model file not found

Train the model first:

```bash
python train.py
```

Verify:

```bash
ls models
```

Expected:

```text
weights.npz
```

---

## License

MIT License
