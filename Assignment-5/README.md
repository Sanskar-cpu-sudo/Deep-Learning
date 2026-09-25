# RNN vs LSTM vs GRU — IMDb Sentiment Classification

## 📌 Overview

This project implements and compares three recurrent neural network architectures — **Simple RNN, LSTM, and GRU** — for binary sentiment classification of movie reviews.

The models are trained on the **IMDb 50K Movie Review Dataset** and evaluated using multiple performance metrics such as accuracy, precision, recall, F1-score, confusion matrix, training time, and parameter count.

---

## 🎯 Objective

The main objectives of this experiment are:

* Implement a **Simple RNN** model.
* Implement an **LSTM** model.
* Implement a **GRU** model.
* Train all three models on IMDb movie reviews.
* Compare their classification performance.
* Analyze training time and number of parameters.
* Evaluate the models using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
  * ROC-AUC
  * Precision-Recall metrics

---

## 📊 Dataset

The project uses the **IMDb Dataset of 50K Movie Reviews**.

The dataset contains:

* **50,000 movie reviews**
* **2 columns**

  * `review` — movie review text
  * `sentiment` — positive or negative
* **25,000 positive reviews**
* **25,000 negative reviews**
* **418 duplicate reviews**
* No missing values were found.

The dataset is downloaded using `kagglehub`.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* SimpleRNN
* LSTM
* GRU
* Embedding

### Data Processing

* NumPy
* Pandas
* Regular Expressions
* Keras Tokenizer
* Sequence Padding

### Machine Learning / Evaluation

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Dataset

* KaggleHub

---

## 🔄 Project Workflow

```text
IMDb Dataset
     ↓
Load Dataset
     ↓
Data Exploration
     ↓
Text Cleaning
     ↓
Train / Validation / Test Split
     ↓
Tokenization
     ↓
Sequence Conversion
     ↓
Padding
     ↓
Embedding
     ↓
┌───────────────┬───────────────┬───────────────┐
│     RNN       │      LSTM     │      GRU      │
└───────────────┴───────────────┴───────────────┘
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Accuracy / Precision / Recall / F1
     ↓
Confusion Matrix
     ↓
Performance Comparison
```

---

## 🧹 Data Preprocessing

The text reviews are processed before being given to the neural networks.

The preprocessing pipeline includes:

1. Loading the IMDb dataset.
2. Cleaning the review text.
3. Converting sentiment labels into numerical values.
4. Tokenizing the text using the Keras `Tokenizer`.
5. Converting reviews into integer sequences.
6. Padding sequences to a fixed length.
7. Feeding the resulting sequences into an embedding layer.

A fixed random seed of **42** is used for reproducibility.

---

## 🧠 Models

All three models use an embedding layer followed by a recurrent layer and a final classification layer.

### 1. Simple RNN

The Simple RNN processes the input sequence sequentially and maintains a hidden state containing information from previous time steps.

```text
Input
  ↓
Embedding
  ↓
SimpleRNN
  ↓
Dropout
  ↓
Dense
  ↓
Binary Classification
```

The RNN uses **64 recurrent units**.

---

### 2. LSTM

LSTM (Long Short-Term Memory) is designed to handle long-term dependencies using memory cells and gates.

```text
Input
  ↓
Embedding
  ↓
LSTM
  ↓
Dropout
  ↓
Dense
  ↓
Binary Classification
```

The LSTM also uses **64 units**.

---

### 3. GRU

GRU (Gated Recurrent Unit) uses update and reset gates to control the flow of information through the sequence.

```text
Input
  ↓
Embedding
  ↓
GRU
  ↓
Dropout
  ↓
Dense
  ↓
Binary Classification
```

The GRU uses **64 units**.

---

## 📈 Model Parameters

| Model | Parameters |
| ----- | ---------: |
| RNN   |  2,572,417 |
| LSTM  |  2,609,473 |
| GRU   |  2,597,313 |

The parameter counts are obtained from the model architectures used in the notebook.

---

## ⏱️ Training Time

| Model |  Training Time |
| ----- | -------------: |
| RNN   | 241.44 seconds |
| LSTM  | 733.08 seconds |
| GRU   | 954.42 seconds |

The recorded training times are from the experiment run in the notebook and can vary depending on the hardware/runtime environment.

---

## 📊 Results

The models were evaluated on the test data using accuracy, precision, recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-Score |
| ----- | -------: | --------: | -----: | -------: |
| RNN   |   51.22% |    51.57% | 40.08% |   45.10% |
| LSTM  |   74.60% |    70.79% | 83.76% |   76.73% |
| GRU   |   88.14% |    85.49% | 91.88% |   88.57% |

These are the recorded test results from the experiment.

---

## 🔍 Evaluation

The experiment evaluates the models using:

### Accuracy

Measures the overall percentage of correctly classified reviews.

### Precision

Measures how many reviews predicted as positive were actually positive.

### Recall

Measures how many actual positive reviews were correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

### Confusion Matrix

The confusion matrices show:

* True Positives
* True Negatives
* False Positives
* False Negatives

Separate confusion matrices are generated for RNN, LSTM, and GRU.

---

## 📉 Visualization

The notebook generates visualizations for comparing the models, including:

* Model performance comparison
* Accuracy comparison
* Confusion matrices
* Training performance
* ROC-related evaluation
* Precision-Recall evaluation

---

## 🏁 Conclusion

This experiment demonstrates the differences between **Simple RNN, LSTM, and GRU** architectures for sequence classification.

The recorded results show substantial differences in classification performance:

* **Simple RNN:** 51.22% accuracy
* **LSTM:** 74.60% accuracy
* **GRU:** 88.14% accuracy

The experiment therefore provides a practical comparison of recurrent architectures on a real-world text classification problem, considering both predictive performance and computational characteristics.

---

## 📁 Project Structure

```text
RNN-LSTM-GRU/
│
├── lab5.ipynb
├── rnn_lstm_gru_results.csv
└── README.md
```

The notebook also saves the final model comparison results into:

```text
rnn_lstm_gru_results.csv
```

---

## ▶️ How to Run

### Option 1 — Google Colab

Open the notebook in Google Colab and execute the cells sequentially.

The notebook includes an **Open in Colab** link for running the experiment directly in Google Colab.

### Option 2 — Local Environment

Install the required libraries:

```bash
pip install kagglehub tensorflow scikit-learn seaborn pandas matplotlib
```

Then open:

```text
lab5.ipynb
```

and run the notebook.

---

## 📚 Learning Outcomes

After completing this experiment, you should understand:

* How recurrent neural networks process sequential data.
* How text is converted into numerical sequences.
* The purpose of an embedding layer.
* The working principle of Simple RNN.
* How LSTM handles long-term dependencies.
* How GRU controls information flow using gates.
* How to evaluate binary classification models.
* How to compare deep learning architectures using multiple metrics.
* The relationship between model architecture, parameters, training time, and performance.

---

## 👨‍💻 Author

**Sanskar**

Deep Learning — RNN, LSTM & GRU Experiment
