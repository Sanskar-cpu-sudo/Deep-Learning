# Lab 7 – Transfer Learning for Cats vs Dogs Classification

## 📌 Overview

This laboratory implements **Transfer Learning** for binary image classification using the **Cats vs Dogs dataset**.

Multiple deep learning architectures are trained and evaluated:

* **AlexNet**
* **VGG16**
* **ResNet50**
* **EfficientNetB0**

The objective is to compare these architectures based on their classification performance and computational requirements.

---

## 🎯 Objective

The main objectives of this experiment are:

* Understand the concept of **Transfer Learning**.
* Apply pretrained CNN architectures to an image-classification problem.
* Perform **Cats vs Dogs classification**.
* Train and evaluate multiple pretrained models.
* Compare models using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Training time
  * Number of parameters
* Visualize and compare model performance.

---

## 📂 Dataset

The experiment uses the **Cats vs Dogs dataset** downloaded from Kaggle.

Dataset source:

**Kaggle – Cats and Dogs Dataset**

The dataset contains two classes:

```text
cats
dogs
```

The notebook downloads the dataset using the Kaggle API and extracts it into the `cats_dogs` directory.

### Dataset Structure

```text
cats_dogs/
│
├── training_set/
│   └── training_set/
│       ├── cats/
│       └── dogs/
│
└── test_set/
    └── test_set/
        ├── cats/
        └── dogs/
```

The dataset contains:

| Dataset           | Images |
| ----------------- | -----: |
| Training dataset  |  8,005 |
| Training subset   |  6,404 |
| Validation subset |  1,601 |
| Test dataset      |  2,023 |

The training data is divided into **80% training and 20% validation**.

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Pandas
* Scikit-learn
* Kaggle API
* Google Colab
* GPU / CUDA acceleration

The notebook was configured to use a **T4 GPU** in Google Colab.

---

## 🔄 Workflow

The overall workflow is:

```text
Kaggle Dataset
       ↓
Download Dataset
       ↓
Extract Dataset
       ↓
Create Training / Validation / Test Sets
       ↓
Resize Images to 224 × 224
       ↓
Apply Transfer Learning
       ↓
Train Multiple CNN Architectures
       ↓
Evaluate Models
       ↓
Calculate Metrics
       ↓
Compare Models
       ↓
Visualize Results
```

---

# 🧠 Transfer Learning

Transfer Learning uses knowledge learned by a model on a large dataset and applies that knowledge to a new task.

Instead of training a deep CNN completely from scratch:

```text
Pretrained CNN
      ↓
Reuse learned features
      ↓
Add classification layer
      ↓
Train for Cats vs Dogs
```

This can reduce training requirements and allow pretrained image features to be reused for the new classification task.

---

# 🏗️ Models Used

## 1. AlexNet

AlexNet is a deep convolutional neural network architecture originally developed for large-scale image classification.

It uses convolutional layers to extract image features followed by fully connected layers for classification.

In this experiment, AlexNet is adapted for the binary:

```text
Cat vs Dog
```

classification task.

---

## 2. VGG16

VGG16 is a CNN architecture consisting of multiple convolutional layers with relatively small **3 × 3 filters**.

General structure:

```text
Input Image
     ↓
Convolution Layers
     ↓
Pooling Layers
     ↓
Feature Extraction
     ↓
Fully Connected Layers
     ↓
Cat / Dog
```

VGG16 is known for its relatively simple and uniform architecture.

---

## 3. ResNet50

ResNet50 is a deep CNN architecture based on **Residual Learning**.

Its important feature is the use of **residual/skip connections**.

Conceptually:

```text
Input
  ↓
Convolution Layers
  ↓
Residual Block
  ├──────────────┐
  ↓              │
More Layers      │
  ↓              │
  └──── + Input ─┘
         ↓
       Output
```

Residual connections help information and gradients flow through deep networks.

---

## 4. EfficientNetB0

EfficientNetB0 is designed to achieve a good balance between:

* Accuracy
* Model size
* Computational cost

It uses an efficient scaling strategy and is suitable when computational efficiency is important.

---

# 🖼️ Image Preprocessing

The images are resized to:

```text
224 × 224
```

and loaded using TensorFlow's:

```python
tf.keras.utils.image_dataset_from_directory()
```

The batch size is:

```python
BATCH_SIZE = 32
```

Binary labels are used because the problem contains only two classes:

```text
0 → cats
1 → dogs
```

The notebook confirms the detected classes as:

```text
['cats', 'dogs']
```

---

# ⚡ Dataset Optimization

TensorFlow's `AUTOTUNE` is used with `prefetch()`:

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)
```

This helps prepare future batches while the model is processing the current batch.

---

# 📊 Evaluation Metrics

The models are compared using several metrics.

### Accuracy

Measures the percentage of correctly classified images.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

### Precision

Measures how many predicted positive samples are actually positive.

$$
Precision = \frac{TP}{TP + FP}
$$

---

### Recall

Measures how many actual positive samples were correctly identified.

$$
Recall = \frac{TP}{TP + FN}
$$

---

### F1-Score

The F1-score combines precision and recall.

$$
F1 = 2 \times \frac{Precision \times Recall}
{Precision + Recall}
$$

---

# 📈 Model Comparison

The notebook evaluates each model and stores:

```text
Model
Accuracy
Precision
Recall
F1 Score
Training Time
Parameters
```

This allows the architectures to be compared not only by classification performance but also by computational requirements.

---

# 📊 Visualizations

The notebook generates visual comparisons for the trained models.

### Accuracy Comparison

```text
Model → Accuracy
```

A bar chart is used to compare test accuracy across the models.

### Precision Comparison

```text
Model → Precision
```

### Recall Comparison

```text
Model → Recall
```

### F1-Score Comparison

```text
Model → F1 Score
```

The notebook also includes plots for model parameter counts and model-performance comparisons.

---

# ⏱️ Training Performance

Training time is recorded for each model.

This is important because a model with high classification performance may also require significantly more computational resources.

The experiment therefore considers both:

```text
Prediction Performance
        +
Computational Cost
```

---

# 🧪 Model Evaluation

For each architecture, predictions are generated and converted into binary class labels using a threshold of `0.5`.

Conceptually:

```python
probability >= 0.5
        ↓
      Dog

probability < 0.5
        ↓
      Cat
```

The resulting predictions are then compared with the actual test labels to calculate the evaluation metrics.

---

# 🏆 Final Comparison

The notebook creates a final comparison based on the calculated metrics.

The comparison considers:

| Metric        | Purpose                              |
| ------------- | ------------------------------------ |
| Accuracy      | Overall classification correctness   |
| Precision     | Correctness of positive predictions  |
| Recall        | Ability to identify positive samples |
| F1 Score      | Balance between precision and recall |
| Training Time | Computational training requirement   |
| Parameters    | Model size/complexity                |

The notebook also creates a final model ranking visualization based on F1-score.

---

# 📁 Project Structure

```text
Lab-7/
│
├── lab7.ipynb
├── README.md
│
└── cats_dogs/
    ├── training_set/
    │   └── training_set/
    │       ├── cats/
    │       └── dogs/
    │
    └── test_set/
        └── test_set/
            ├── cats/
            └── dogs/
```

---

# ▶️ How to Run

### 1. Open the notebook

The notebook can be executed using Google Colab.

### 2. Enable GPU

In Google Colab:

```text
Runtime
   ↓
Change runtime type
   ↓
GPU
```

The notebook detects the available GPU using TensorFlow.

### 3. Install Kaggle

```python
!pip install -q kaggle
```

### 4. Download the dataset

```python
!kaggle datasets download -d tongpython/cat-and-dog
```

### 5. Extract the dataset

```python
import zipfile

with zipfile.ZipFile("cat-and-dog.zip", "r") as zip_ref:
    zip_ref.extractall("cats_dogs")
```

### 6. Run the notebook

Execute the cells sequentially to:

* Load the dataset
* Preprocess images
* Build the models
* Train the models
* Evaluate the models
* Generate comparison graphs

---

# 💡 Key Learning Outcomes

After completing this experiment, the following concepts are demonstrated:

* Understanding **Transfer Learning**
* Using pretrained CNN architectures
* Image classification using TensorFlow/Keras
* Dataset splitting and preprocessing
* GPU-based deep learning
* Model evaluation using multiple metrics
* Comparing CNN architectures
* Understanding model parameters and training time
* Visualizing model performance

---

# 🔑 Key Concepts

```text
Transfer Learning
       ↓
Pretrained CNN
       ↓
Feature Extraction
       ↓
Fine-tuning / Classification
       ↓
Binary Classification
       ↓
Model Evaluation
       ↓
Performance Comparison
```

---

# 📌 Conclusion

This experiment demonstrates how **Transfer Learning** can be applied to the Cats vs Dogs image-classification problem using multiple CNN architectures.

AlexNet, VGG16, ResNet50, and EfficientNetB0 are evaluated using common classification metrics along with training time and parameter count. This provides a practical comparison of different deep-learning architectures for image classification.

The experiment demonstrates that model evaluation should consider **both predictive performance and computational requirements**, rather than relying on a single metric.

---

## 👨‍💻 Author

**Sanskar**

B.Tech – Computer Engineering / Artificial Intelligence

Vishwakarma Institute of Technology, Pune
