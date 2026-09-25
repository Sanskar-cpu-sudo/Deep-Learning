# Lab 1 – MNIST Handwritten Digit Classification

## 📌 Overview

This laboratory implements a **Neural Network using TensorFlow and Keras** to recognize handwritten digits from the **MNIST dataset**.

The model is trained using handwritten digit images and then evaluated on unseen test images to determine its classification performance.

The complete workflow includes:

* Loading the MNIST dataset
* Preprocessing image data
* Normalizing pixel values
* Building a neural network
* Training the model
* Evaluating the model
* Testing predictions
* Visualizing handwritten digits

---

## 🎯 Objective

The main objective of this experiment is to understand the basic workflow of building a neural network for image classification.

The experiment demonstrates how TensorFlow and Keras can be used to build, train, evaluate, and test a neural network for recognizing handwritten digits.

---

## 📂 Dataset

The experiment uses the **MNIST handwritten digit dataset**.

MNIST contains grayscale images representing digits from:

```text
0 → 9
```

Each image has a resolution of:

```text
28 × 28 pixels
```

### Dataset Size

| Dataset      | Number of Images |
| ------------ | ---------------: |
| Training Set |           60,000 |
| Test Set     |           10,000 |

The notebook loads the dataset using:

```python
tf.keras.datasets.mnist.load_data()
```

The recorded shapes are:

```text
Training images : (60000, 28, 28)
Test images     : (10000, 28, 28)
```

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Matplotlib
* NumPy
* Google Colab / Jupyter Notebook

---

# 🔄 Workflow

The complete workflow is:

```text
MNIST Dataset
      ↓
Load Training & Test Data
      ↓
Normalize Pixel Values
      ↓
Build Neural Network
      ↓
Compile Model
      ↓
Train Model
      ↓
Evaluate on Test Data
      ↓
Generate Predictions
      ↓
Visualize Results
```

---

# 🖼️ Understanding the Input

Each MNIST image is:

```text
28 × 28
```

Since the images are grayscale, each pixel contains an intensity value between:

```text
0 → 255
```

where:

```text
0   → Black
255 → White
```

The notebook scales these values to the range:

```text
0 → 1
```

using:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

This normalization makes the input values easier for the neural network to process.

---

# 🧠 Neural Network

The experiment uses a neural network implemented using **TensorFlow/Keras**.

The general classification process is:

```text
28 × 28 Image
      ↓
Preprocessing
      ↓
Neural Network
      ↓
Feature Learning
      ↓
Output Layer
      ↓
10 Digit Classes
```

The output represents the ten possible digit classes:

```text
0  1  2  3  4
5  6  7  8  9
```

---

# 🔢 Classification Problem

This is a **multi-class classification** problem.

There are 10 possible classes:

```text
Class 0 → Digit 0
Class 1 → Digit 1
Class 2 → Digit 2
Class 3 → Digit 3
Class 4 → Digit 4
Class 5 → Digit 5
Class 6 → Digit 6
Class 7 → Digit 7
Class 8 → Digit 8
Class 9 → Digit 9
```

For each input image, the neural network predicts which digit the image represents.

---

# 📊 Data Preprocessing

The preprocessing performed in the notebook includes:

### 1. Dataset Loading

```python
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.mnist.load_data()
```

### 2. Pixel Normalization

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

This converts the original pixel range:

```text
0 – 255
```

into:

```text
0 – 1
```

---

# 👀 Data Visualization

The notebook also displays MNIST images using Matplotlib.

This allows the handwritten digits in the dataset to be visually inspected before training the model.

Example:

```text
       ███
      █   █
          █
         █
        █
       █
      █
```

The actual MNIST samples are displayed directly from the dataset.

---

# 🧪 Model Training

The neural network is trained using:

```text
Training Images
      +
Training Labels
      ↓
Neural Network
      ↓
Learn Digit Patterns
```

During training, the model learns visual features that help distinguish one handwritten digit from another.

For example, it learns patterns associated with:

```text
0 → circular structure
1 → vertical structure
8 → two-loop structure
```

and other digit-specific patterns.

---

# 📈 Model Evaluation

After training, the model is evaluated using the test dataset.

The purpose of the test set is to measure how well the trained network performs on images that it has not seen during training.

Conceptually:

```text
Training Data
     ↓
Model learns
     ↓
Test Data
     ↓
Predictions
     ↓
Performance Evaluation
```

---

# 🔮 Predictions

The trained model can be used to predict the digit represented by an unseen MNIST image.

The prediction workflow is:

```text
Test Image
    ↓
Normalize Image
    ↓
Pass Through Neural Network
    ↓
Calculate Class Probabilities
    ↓
Select Predicted Class
    ↓
Predicted Digit
```

For example:

```text
Actual Digit:    5
Predicted Digit: 5
```

---

# 📊 Performance Metrics

The primary metric for this classification task is **accuracy**.

### Accuracy

Accuracy represents the percentage of test images classified correctly.

$$
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\times 100
$$

For example, if:

```text
9,500 / 10,000
```

images are correctly classified:

```text
Accuracy = 95%
```

---

# 📁 Project Structure

```text
Lab-1/
│
├── lab1.ipynb
└── README.md
```

The notebook is also configured with a Google Colab link for execution.

---

# ▶️ How to Run

## 1. Open the Notebook

Open:

```text
lab1.ipynb
```

in:

* Google Colab
* Jupyter Notebook
* VS Code with Jupyter support

## 2. Install Dependencies

If TensorFlow is not already installed:

```bash
pip install tensorflow matplotlib numpy
```

## 3. Run the Notebook

Execute the cells sequentially.

The notebook will:

1. Import TensorFlow
2. Download MNIST
3. Preprocess the images
4. Build the neural network
5. Train the model
6. Evaluate the model
7. Generate predictions
8. Display results

---

# ⚠️ Environment Note

The notebook contains an earlier local execution attempt where importing TensorFlow resulted in an `ImportError: initialization failed`. The notebook subsequently proceeds with the TensorFlow/Keras workflow and successfully downloads the MNIST dataset.

For reproducibility, Google Colab can be used through the notebook's Colab link.

---

# 💡 Key Learning Outcomes

After completing this experiment, the following concepts are demonstrated:

* Understanding the MNIST dataset
* Loading datasets using TensorFlow/Keras
* Image preprocessing
* Pixel normalization
* Neural network-based image classification
* Multi-class classification
* Model training
* Model evaluation
* Generating predictions
* Visualizing image data

---

# 🔑 Key Concepts

```text
MNIST
  ↓
Image Preprocessing
  ↓
Normalization
  ↓
Neural Network
  ↓
Training
  ↓
Testing
  ↓
Prediction
  ↓
Accuracy
```

---

# 📌 Conclusion

This laboratory demonstrates the complete process of building a neural-network-based image classification system using the **MNIST handwritten digit dataset**.

The images are normalized from their original pixel range of **0–255** to **0–1**, after which the neural network learns patterns from the training data and predicts digit classes for unseen test images.

The experiment provides a foundation for understanding how neural networks are applied to image classification problems using TensorFlow and Keras.

---

## 👨‍💻 Author

**Sanskar**

B.Tech – Computer Engineering / Artificial Intelligence

Vishwakarma Institute of Technology, Pune
