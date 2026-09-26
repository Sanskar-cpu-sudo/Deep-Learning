# BERT for Sentiment Analysis

## 📌 Overview

This project performs **binary sentiment classification** on movie reviews using a pre-trained **BERT (`bert-base-uncased`)** model.

The model is fine-tuned on a sample of the **IMDB movie review dataset** to classify reviews as:

* `0` → Negative
* `1` → Positive

The project uses Hugging Face Transformers and evaluates the model using **Accuracy, Precision, Recall, and F1-score**.

---

## 🎯 Objectives

* Load and preprocess the IMDB sentiment dataset.
* Use a pre-trained BERT tokenizer to convert text into tokens.
* Fine-tune BERT for binary sentiment classification.
* Evaluate model performance using standard classification metrics.
* Test the trained model on new movie-review sentences.

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **Hugging Face Datasets**
* **Scikit-learn**
* **NumPy**
* **CUDA / GPU** for accelerated training

---

## 📂 Dataset

The project uses the **IMDB Movie Review Dataset** available through Hugging Face:

```text
stanfordnlp/imdb
```

The notebook uses a smaller sample for faster execution:

```text
Training samples: 2000
Testing samples : 500
```

Each review contains a binary sentiment label:

```text
0 → Negative
1 → Positive
```

---

## 🔄 Workflow

```text
IMDB Dataset
      ↓
Select Training & Testing Samples
      ↓
BERT Tokenization
      ↓
Pre-trained BERT Model
      ↓
Fine-tuning
      ↓
Sentiment Prediction
      ↓
Model Evaluation
      ↓
Test on New Reviews
```

---

## ⚙️ Implementation Steps

### 1. Install Dependencies

The required libraries are installed using:

```bash
pip install -U transformers datasets accelerate scikit-learn
```

### 2. Load Dataset

The IMDB dataset is loaded using Hugging Face Datasets and shuffled with a fixed random seed.

A sample of 2000 training reviews and 500 testing reviews is used.

### 3. Tokenization

The pre-trained BERT tokenizer is loaded:

```python
MODEL_NAME = "bert-base-uncased"
```

Reviews are tokenized with:

```text
max_length = 256
truncation = True
```

Dynamic padding is performed using `DataCollatorWithPadding`.

### 4. Load BERT Model

`AutoModelForSequenceClassification` is used with two output classes:

```text
NEGATIVE → 0
POSITIVE → 1
```

The pre-trained BERT model is fine-tuned for the sentiment classification task.

### 5. Fine-Tuning

The model is trained using Hugging Face's `Trainer`.

Main training parameters:

| Parameter             | Value |
| --------------------- | ----: |
| Epochs                |     1 |
| Training Batch Size   |     8 |
| Evaluation Batch Size |    16 |
| Learning Rate         |  2e-5 |
| Weight Decay          |  0.01 |
| Max Sequence Length   |   256 |
| Seed                  |    42 |

GPU mixed-precision (`fp16`) is enabled when CUDA is available.

### 6. Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

The evaluation metrics are calculated using Scikit-learn.

---

## 📊 Results

After one epoch of fine-tuning on the selected sample, the notebook achieved:

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **87.20%** |
| Precision       | **84.73%** |
| Recall          | **90.24%** |
| F1-Score        | **87.40%** |
| Validation Loss | **0.2912** |

These results are obtained using the **2000 training samples and 500 testing samples** used in the notebook.

---

## 🧪 Example

The trained model can be used to classify new movie reviews.

Example:

```text
"I really enjoyed this movie!"
```

Possible prediction:

```text
POSITIVE
```

Similarly, a review expressing strong dislike can be classified as:

```text
NEGATIVE
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install dependencies

```bash
pip install -U transformers datasets accelerate scikit-learn
```

PyTorch should also be installed according to your system's CPU/GPU configuration.

### 3. Open the notebook

```text
Assignemt_8.ipynb
```

Run the notebook cells sequentially.

### 4. GPU Recommendation

Training is significantly faster with a CUDA-enabled GPU.

The notebook automatically checks for CUDA:

```python
torch.cuda.is_available()
```

If available, it uses the GPU for training.

---

## 📁 Project Structure

```text
.
├── Assignemt_8.ipynb
└── README.md
```

---

## 🔑 Key Concepts

* Transfer Learning
* BERT
* Natural Language Processing
* Text Tokenization
* Sequence Classification
* Fine-Tuning
* Sentiment Analysis
* Precision
* Recall
* F1-Score
* Hugging Face Trainer

---

## 📌 Conclusion

This project demonstrates how a **pre-trained BERT model can be fine-tuned for sentiment analysis**. Using a relatively small sample of the IMDB dataset, the model achieved **87.20% accuracy and 87.40% F1-score** after one training epoch.

The approach can be extended by using more training data, increasing the number of epochs, and tuning hyperparameters to improve performance.
