# CNN-Based Tomato Disease Classification

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** for classifying tomato plant diseases from leaf images.

The model is trained using the **Tomato Disease dataset** and evaluated using validation accuracy, validation loss, sample predictions, and visual comparison between actual and predicted classes.

---

## 🎯 Objective

The main objective of this experiment is to:

* Implement a CNN for image classification.
* Classify tomato leaves into different disease categories.
* Train the CNN using training images.
* Validate the model using a validation dataset.
* Analyze training and validation performance.
* Generate predictions for sample images.
* Visualize actual and predicted disease classes.

---

## 📊 Dataset

The project uses a **Tomato Disease dataset** organized into separate training and validation directories.

```text
Tomato_Disease/
│
├── train/
│   ├── Tomato___Bacterial_spot
│   ├── Tomato___Early_blight
│   ├── Tomato___healthy
│   ├── Tomato___Late_blight
│   ├── Tomato___Leaf_Mold
│   ├── Tomato___Septoria_leaf_spot
│   ├── Tomato___Spider_mites Two-spotted_spider_mite
│   ├── Tomato___Target_Spot
│   ├── Tomato___Tomato_mosaic_virus
│   └── Tomato___Tomato_Yellow_Leaf_Curl_Virus
│
└── val/
    └── Same 10 classes
```

The dataset contains:

* **10,000 training images**
* **1,000 validation images**
* **10 disease/health classes**

---

## 🌱 Classes

The CNN classifies images into the following 10 categories:

| Class | Disease / Category            |
| ----: | ----------------------------- |
|     0 | Tomato Bacterial Spot         |
|     1 | Tomato Early Blight           |
|     2 | Tomato Late Blight            |
|     3 | Tomato Leaf Mold              |
|     4 | Tomato Septoria Leaf Spot     |
|     5 | Tomato Spider Mites           |
|     6 | Tomato Target Spot            |
|     7 | Tomato Yellow Leaf Curl Virus |
|     8 | Tomato Mosaic Virus           |
|     9 | Tomato Healthy                |

The notebook detects **10 classes** from the directory structure.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Network (CNN)

### Libraries

* NumPy
* Matplotlib
* TensorFlow

---

## 🔄 Project Workflow

```text
Tomato Disease Dataset
        ↓
Load Training & Validation Images
        ↓
Resize Images to 224 × 224
        ↓
Create Batches
        ↓
CNN Model
        ↓
Convolution
        ↓
Max Pooling
        ↓
Convolution
        ↓
Max Pooling
        ↓
Convolution
        ↓
Max Pooling
        ↓
Convolution
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense Layer
        ↓
Dropout
        ↓
Softmax Output
        ↓
Disease Classification
        ↓
Validation & Prediction
```

---

## 🖼️ Image Preprocessing

Images are loaded using TensorFlow's `image_dataset_from_directory()`.

The images are resized to:

```text
224 × 224 pixels
```

The batch size is:

```text
32
```

Training images are shuffled, while validation images are not shuffled.

---

## 🧠 CNN Architecture

The model is implemented using the Keras Sequential API.

### Architecture

```text
Input: 224 × 224 × 3
        ↓
Conv2D — 32 filters, 3 × 3
        ↓
MaxPooling2D — 2 × 2
        ↓
Conv2D — 64 filters, 3 × 3
        ↓
MaxPooling2D — 2 × 2
        ↓
Conv2D — 128 filters, 3 × 3
        ↓
MaxPooling2D — 2 × 2
        ↓
Conv2D — 128 filters, 3 × 3
        ↓
MaxPooling2D — 2 × 2
        ↓
Flatten
        ↓
Dense — 128 neurons
        ↓
Dropout — 0.5
        ↓
Dense — 10 neurons
        ↓
Softmax
```

The CNN contains four convolutional blocks followed by flattening, a dense layer, dropout, and a 10-class softmax output layer.

---

## 🔍 CNN Layers

### 1. Convolutional Layers

Four `Conv2D` layers are used:

```text
32 filters
64 filters
128 filters
128 filters
```

The convolution layers use:

```text
Kernel = 3 × 3
Activation = ReLU
```

These layers learn visual features from the tomato leaf images.

---

### 2. Max Pooling

Each convolutional layer is followed by:

```text
MaxPooling2D(2 × 2)
```

Max pooling reduces the spatial dimensions of the feature maps while retaining important features.

---

### 3. Flatten

After the convolution and pooling layers, the feature maps are converted into a one-dimensional vector using:

```python
Flatten()
```

The resulting feature vector contains **18,432 values**.

---

### 4. Dense Layer

The flattened features are passed to:

```text
Dense(128, activation="relu")
```

This layer performs high-level feature learning before classification.

---

### 5. Dropout

A dropout rate of:

```text
0.5
```

is used to reduce overfitting during training.

---

### 6. Output Layer

The final layer contains:

```text
10 neurons
```

with a `softmax` activation function.

Each neuron represents one tomato disease/health class.

---

## ⚙️ Model Configuration

The model is compiled using:

```text
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metric: Accuracy
```

The configuration is suitable for multi-class classification with integer class labels.

---

## 📐 Model Parameters

The CNN contains:

```text
Total Parameters: 2,601,546
Trainable Parameters: 2,601,546
Non-trainable Parameters: 0
```

The model occupies approximately **9.92 MB** according to the model summary.

### Parameter Distribution

| Layer             |    Parameters |
| ----------------- | ------------: |
| Conv2D — 32       |           896 |
| Conv2D — 64       |        18,496 |
| Conv2D — 128      |        73,856 |
| Conv2D — 128      |       147,584 |
| Dense — 128       |     2,359,424 |
| Output Dense — 10 |         1,290 |
| **Total**         | **2,601,546** |

---

## 🏋️ Training

The model is trained for:

```text
3 epochs
```

using the training dataset and validated using the validation dataset.

### Training Results

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
| ----: | ----------------: | ------------: | ------------------: | --------------: |
|     1 |            12.89% |        2.3358 |              14.40% |          2.2730 |
|     2 |            12.39% |        2.2791 |              12.40% |          2.2824 |
|     3 |            11.86% |        2.2921 |              12.90% |          2.2770 |

These values are the recorded results from the notebook run.

---

## 📊 Validation Performance

The final validation evaluation produced:

```text
Validation Loss:     2.27701997756958
Validation Accuracy: 0.1289999932050705
```

Therefore, the recorded validation accuracy is approximately:

```text
12.90%
```

---

## 🔮 Predictions

The trained model is also used to generate predictions for images from the validation dataset.

The predicted class is obtained using:

```python
np.argmax(predictions, axis=1)
```

The notebook prints the actual and predicted class for sample images.

Example output includes:

```text
Actual: Tomato___Bacterial_spot
Predicted: Tomato___Bacterial_spot
```

and cases where the model predicts a different class, such as:

```text
Actual: Tomato___Bacterial_spot
Predicted: Tomato___Target_Spot
```

---

## 🖼️ Prediction Visualization

The notebook visualizes **9 validation images** in a `3 × 3` grid.

Each image displays:

```text
Actual: <class>
Predicted: <class>
```

This provides a visual way to inspect how the CNN is classifying individual tomato leaf images.

---

## 📉 Evaluation

The experiment evaluates the CNN using:

### Accuracy

Measures the proportion of correctly classified images.

### Loss

Measures the difference between the model's predicted probabilities and the actual class labels.

### Predictions

The model's predicted disease class is compared with the actual class.

### Visualization

Sample predictions are displayed alongside the original images for qualitative inspection.

---

## 📁 Project Structure

```text
Tomato-Disease-CNN/
│
├── lab6.ipynb
├── Tomato_Disease/
│   ├── train/
│   │   ├── Tomato___Bacterial_spot/
│   │   ├── Tomato___Early_blight/
│   │   ├── Tomato___healthy/
│   │   ├── Tomato___Late_blight/
│   │   ├── Tomato___Leaf_Mold/
│   │   ├── Tomato___Septoria_leaf_spot/
│   │   ├── Tomato___Spider_mites Two-spotted_spider_mite/
│   │   ├── Tomato___Target_Spot/
│   │   ├── Tomato___Tomato_mosaic_virus/
│   │   └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   │
│   └── val/
│       └── Same 10 classes
│
└── README.md
```

---

## ▶️ How to Run

### Option 1 — Google Colab

Open the notebook in Google Colab and execute the cells sequentially.

The notebook includes an **Open in Colab** button.

### Option 2 — Local Environment

Install TensorFlow and the required libraries:

```bash
pip install tensorflow numpy matplotlib
```

Then open:

```text
lab6.ipynb
```

Make sure the dataset is available at:

```text
./Tomato_Disease/
```

with the `train` and `val` directories.

---

## ⚠️ Note About the Recorded Results

The recorded experiment achieved approximately **12.9% validation accuracy after 3 epochs**.

Since there are **10 classes**, this result indicates that the model run did not learn the classification task effectively in the recorded experiment. This README reports the notebook's actual output rather than replacing it with an expected or improved result.

The notebook also recorded a TensorFlow warning indicating that native Windows TensorFlow GPU support was unavailable in the environment used for the run.

---

## 🎓 Learning Outcomes

After completing this experiment, you should understand:

* How CNNs are used for image classification.
* How images are loaded using directory-based datasets.
* The role of convolutional layers.
* The purpose of max pooling.
* How feature maps are converted using flattening.
* The role of dense layers in classification.
* How dropout helps reduce overfitting.
* How softmax performs multi-class classification.
* How to train and validate a CNN.
* How to analyze model predictions visually.

---

## 👨‍💻 Author

**Sanskar**

Deep Learning — CNN-Based Tomato Disease Classification
