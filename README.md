# Deep Learning Laboratory Assignments

This repository contains implementations of various **Deep Learning laboratory assignments** covering fundamental neural networks, sequence models, convolutional networks, transfer learning, transformers, computer vision, NLP, time-series forecasting, and audio processing.

The assignments are implemented primarily using **Python, TensorFlow, Keras, Scikit-learn, Hugging Face Transformers, and Google Colab**.

---

## 📌 Assignments

| No. | Assignment                                  | Key Concepts                                                        |
| --- | ------------------------------------------- | ------------------------------------------------------------------- |
| 1   | TensorFlow/Keras Setup & Data Preprocessing | TensorFlow, Keras, normalization, train-test split, visualization   |
| 2   | Multilayer Perceptron Classification        | MLP, classification, Iris/Wine dataset, confusion matrix            |
| 3   | Forward & Backpropagation                   | Learning rate, epochs, gradient descent, model optimization         |
| 4   | LSTM Time-Series Forecasting                | LSTM, sequences, forecasting, stock/weather/sales data              |
| 5   | RNN vs LSTM vs GRU                          | Sequence classification, recurrent networks, performance comparison |
| 6   | CNN Image Classification                    | CNN, image preprocessing, Tomato/Soybean disease classification     |
| 7   | Transfer Learning                           | AlexNet, VGG16, ResNet50, EfficientNetB0                            |
| 8   | BERT Text Classification                    | Transformers, BERT, sentiment analysis, NLP                         |
| 9   | Vision Transformer                          | ViT, image classification, CNN vs Transformer                       |
| 10  | Audio Classification with STT & TTS         | Audio processing, Speech-to-Text, Text-to-Speech, deep learning     |

---

# Assignment 1: TensorFlow/Keras Setup and Data Preprocessing

## Objective

Install and configure **TensorFlow/Keras in Google Colab** and perform common data preprocessing operations on a sample dataset.

## Tasks

* Install and import TensorFlow/Keras
* Load a sample dataset
* Explore dataset features
* Handle missing values if required
* Perform feature normalization
* Split the dataset into training and testing sets
* Visualize the dataset
* Prepare data for training a neural network

## Concepts Covered

* TensorFlow
* Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Data preprocessing
* Normalization
* Train-test splitting

---

# Assignment 2: Multilayer Perceptron for Classification

## Objective

Design and implement a **Multilayer Perceptron (MLP)** for classifying samples from the **Iris or Wine dataset**.

## Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Input Layer
   ↓
Hidden Dense Layers
   ↓
Output Layer
   ↓
Prediction
   ↓
Evaluation
```

## Evaluation Metrics

* Accuracy
* Confusion Matrix

## Concepts Covered

* Artificial Neural Networks
* Multilayer Perceptron
* Dense layers
* Activation functions
* Softmax
* Classification

---

# Assignment 3: Forward Propagation and Backpropagation

## Objective

Implement neural network training using **forward propagation and backpropagation** with TensorFlow/Keras.

The experiment also studies how different hyperparameters affect model performance.

## Parameters Analyzed

### Learning Rate

Different learning rates can be tested, such as:

```text
0.0001
0.001
0.01
0.1
```

### Number of Epochs

Example:

```text
10
25
50
100
```

## Basic Training Flow

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Calculation
  ↓
Backpropagation
  ↓
Gradient Calculation
  ↓
Weight Update
  ↓
Next Epoch
```

## Analysis

The experiment demonstrates the impact of:

* Small learning rate
* Large learning rate
* Undertraining
* Overtraining
* Number of epochs
* Convergence of the loss function

---

# Assignment 4: LSTM for Time-Series Forecasting

## Objective

Develop an **LSTM-based deep learning model** for forecasting sequential/time-series data.

Possible datasets include:

* Stock prices
* Weather data
* Sales data

## Architecture

```text
Time-Series Dataset
        ↓
Data Cleaning
        ↓
Normalization
        ↓
Sequence Creation
        ↓
LSTM Layer
        ↓
Dense Layer
        ↓
Prediction
        ↓
Actual vs Predicted Comparison
```

## Evaluation Metrics

Possible metrics:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

## Concepts Covered

* Time-series data
* Sequential modeling
* LSTM
* Sliding window
* Forecasting
* Sequence preprocessing

---

# Assignment 5: Comparison of RNN, LSTM, and GRU

## Objective

Implement and compare three popular recurrent neural network architectures:

1. Simple RNN
2. LSTM
3. GRU

for a **sequence classification problem**.

## Architectures

### Simple RNN

```text
Embedding
   ↓
SimpleRNN
   ↓
Dense
   ↓
Output
```

### LSTM

```text
Embedding
   ↓
LSTM
   ↓
Dense
   ↓
Output
```

### GRU

```text
Embedding
   ↓
GRU
   ↓
Dense
   ↓
Output
```

## Comparison Criteria

Models can be compared based on:

* Accuracy
* Precision
* Recall
* F1-score
* Training time
* Validation loss
* Number of parameters

## Key Observation

Simple RNNs are computationally simple but can suffer from **vanishing gradients** when handling long sequences.

LSTM and GRU use gating mechanisms to retain important information for longer periods.

---

# Assignment 6: CNN for Image Classification

## Objective

Design and implement a **Convolutional Neural Network (CNN)** for plant disease image classification.

Possible datasets:

* Tomato Disease Dataset
* Soybean Disease Dataset

## CNN Architecture

```text
Input Image
     ↓
Convolution Layer
     ↓
ReLU
     ↓
Max Pooling
     ↓
Convolution Layer
     ↓
Max Pooling
     ↓
Flatten
     ↓
Dense Layer
     ↓
Softmax
     ↓
Disease Class
```

## Image Processing

The following operations may be performed:

* Image resizing
* Pixel normalization
* Data augmentation
* Training-validation splitting

## Evaluation

* Accuracy
* Loss
* Confusion Matrix
* Classification Report

---

# Assignment 7: Transfer Learning

## Objective

Implement **transfer learning** using pre-trained deep learning models and compare their performance for image classification.

## Models

### AlexNet

One of the early influential deep CNN architectures designed for image classification.

### VGG16

Uses multiple small `3 × 3` convolution filters with a deep architecture.

### ResNet50

Introduces **residual connections** to solve degradation and vanishing-gradient problems in deep networks.

### EfficientNetB0

Uses **compound scaling** to efficiently balance:

* Network depth
* Network width
* Image resolution

## Transfer Learning Flow

```text
Image Dataset
      ↓
Image Preprocessing
      ↓
Pre-trained Model
      ↓
Freeze Base Layers
      ↓
Custom Classification Head
      ↓
Training
      ↓
Fine-Tuning
      ↓
Evaluation
```

## Comparison

| Model          |          Accuracy |     Training Time |        Parameters |
| -------------- | ----------------: | ----------------: | ----------------: |
| AlexNet        | Experiment Result | Experiment Result | Experiment Result |
| VGG16          | Experiment Result | Experiment Result | Experiment Result |
| ResNet50       | Experiment Result | Experiment Result | Experiment Result |
| EfficientNetB0 | Experiment Result | Experiment Result | Experiment Result |

---

# Assignment 8: BERT for Sentiment Analysis

## Objective

Implement a **pre-trained BERT model** for sentiment analysis or text classification.

## Workflow

```text
Raw Text
   ↓
BERT Tokenizer
   ↓
Input IDs + Attention Mask
   ↓
Pre-trained BERT
   ↓
Classification Layer
   ↓
Sentiment / Text Class
```

## Example Task

Sentiment classification:

```text
"This product is excellent!"
        ↓
      BERT
        ↓
     Positive
```

## Concepts Covered

* Natural Language Processing
* Transformers
* Attention mechanism
* BERT
* Tokenization
* Fine-tuning
* Text classification

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# Assignment 9: Vision Transformer for Image Classification

## Objective

Perform image classification using a pre-trained **Vision Transformer (ViT)** and compare its performance with a traditional CNN.

## Vision Transformer Pipeline

```text
Input Image
     ↓
Image Patches
     ↓
Patch Embeddings
     ↓
Positional Embeddings
     ↓
Transformer Encoder
     ↓
Classification Head
     ↓
Predicted Class
```

## CNN vs ViT

| Feature                  | CNN           | Vision Transformer        |
| ------------------------ | ------------- | ------------------------- |
| Main operation           | Convolution   | Self-Attention            |
| Image representation     | Local regions | Image patches             |
| Local feature extraction | Strong        | Learned through attention |
| Global relationships     | Indirect      | Direct                    |
| Data requirement         | Moderate      | Often larger              |
| Pre-trained performance  | High          | Very High                 |

## Evaluation

Both models can be compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Training time

---

# Assignment 10: Audio Classification with STT and TTS

## Objective

Design and implement a deep learning application for **audio classification** incorporating:

* Audio classification
* Speech-to-Text (STT)
* Text-to-Speech (TTS)

## System Architecture

```text
                    Audio Input
                         ↓
                  Audio Preprocessing
                         ↓
                 Feature Extraction
                  (MFCC / Mel-Spectrogram)
                         ↓
                Deep Learning Model
                (CNN / RNN / LSTM)
                         ↓
                 Audio Classification
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
       Speech-to-Text          Predicted Class
            (STT)
              ↓
         Generated Text
              ↓
       Text-to-Speech
            (TTS)
              ↓
         Audio Output
```

## Audio Features

Common features include:

* MFCC
* Mel-Spectrogram
* Chroma Features
* Spectral Centroid
* Zero Crossing Rate

## Evaluation Metrics

For classification:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

For speech recognition:

* Word Error Rate (WER)

---

# Technologies Used

## Programming Language

* Python

## Deep Learning

* TensorFlow
* Keras

## Machine Learning

* Scikit-learn

## Data Processing

* NumPy
* Pandas

## Visualization

* Matplotlib

## NLP

* Hugging Face Transformers
* BERT

## Computer Vision

* OpenCV
* TensorFlow/Keras
* Vision Transformer

## Audio Processing

* Librosa
* Speech Recognition libraries
* Text-to-Speech libraries

## Development Environment

* Google Colab
* Jupyter Notebook



# Installation

Most experiments are designed to run directly in **Google Colab**.

For local execution, clone the repository:

```bash
git clone <your-repository-url>
cd Deep-Learning-Lab
```

Install the required Python packages:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

For NLP experiments:

```bash
pip install transformers datasets
```

For computer vision experiments:

```bash
pip install opencv-python pillow
```

For audio processing:

```bash
pip install librosa soundfile SpeechRecognition
```

---

# Running the Notebooks

### Google Colab

1. Open the required `.ipynb` file.
2. Upload it to Google Colab.
3. Select:

```text
Runtime → Change Runtime Type → T4 GPU
```

4. Install required dependencies.
5. Upload/connect the dataset.
6. Run all cells sequentially.

---

# Learning Outcomes

After completing these assignments, the following deep learning concepts are explored:

* Data preprocessing for deep learning
* Neural network architecture design
* Forward propagation
* Backpropagation
* Gradient descent
* Hyperparameter tuning
* Multilayer Perceptrons
* Recurrent Neural Networks
* LSTM
* GRU
* Convolutional Neural Networks
* Transfer Learning
* Transformer architecture
* BERT
* Vision Transformers
* Time-series forecasting
* Audio classification
* Speech-to-Text
* Text-to-Speech
* Model evaluation

---

# Model Evaluation

Different assignments use different evaluation metrics depending on the problem.

### Classification

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
```

### Regression / Time-Series Forecasting

```text
MAE
MSE
RMSE
```

### Speech Recognition

```text
Word Error Rate (WER)
```

---

# Author

**Sahil Patil**

B.Tech — Computer Science and Engineering
Specialization: Artificial Intelligence
Vishwakarma Institute of Technology, Pune

---

# Disclaimer

This repository is created for **academic and educational purposes** as part of Deep Learning laboratory assignments.

Datasets and pre-trained models used in the experiments belong to their respective owners and are used only for learning and experimentation.
