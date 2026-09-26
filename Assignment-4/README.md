# Assignment 4 — LSTM-Based Time-Series Forecasting

## 📌 Overview

This project develops an **LSTM (Long Short-Term Memory) based model for time-series forecasting** using the **Airline Passengers dataset**.

The main objective is to study how different amounts of historical information affect forecasting performance. Three different look-back windows are compared:

* **6 months** — short-term historical information
* **12 months** — one complete yearly seasonal cycle
* **24 months** — two complete yearly seasonal cycles

The models are evaluated using **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**.

---

## 🎯 Problem Statement

**Develop an LSTM-based model for time-series forecasting using stock-price, weather, or sales datasets.**

For this implementation, the **Airline Passengers dataset** is used to forecast future passenger numbers based on historical passenger data.

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📊 Dataset

The project uses the **Airline Passengers dataset**, containing monthly passenger numbers.

Dataset characteristics:

* **144 observations**
* **2 columns**

  * `Month`
  * `Passengers`
* Monthly data
* No missing values

Example:

| Month   | Passengers |
| ------- | ---------: |
| 1949-01 |        112 |
| 1949-02 |        118 |
| 1949-03 |        132 |
| 1949-04 |        129 |
| 1949-05 |        121 |

The dataset is automatically downloaded if it is not already available locally.

---

## 🔄 Project Workflow

```text
Airline Passenger Dataset
          ↓
Data Loading
          ↓
Data Cleaning & Validation
          ↓
Time-Series Visualization
          ↓
Train/Test Split
          ↓
Min-Max Scaling
          ↓
Create Sequential Training Windows
          ↓
LSTM Model
          ↓
Model Training
          ↓
Prediction
          ↓
Inverse Scaling
          ↓
MAE & RMSE Evaluation
          ↓
Compare 6, 12 & 24 Month Windows
```

---

## 🧠 LSTM Model

LSTM is used because it is designed to learn patterns and dependencies from sequential data.

The model uses historical passenger values to predict the next value in the time series.

Different historical window sizes are tested:

### 6-Month Window

The model uses the previous **6 months** to predict the next month's passenger count.

### 12-Month Window

The model uses the previous **12 months**, representing one complete yearly cycle.

### 24-Month Window

The model uses the previous **24 months**, representing two yearly cycles.

This allows the effect of different historical contexts on forecasting performance to be compared.

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset.
2. Convert the `Month` column into datetime format.
3. Sort the data chronologically.
4. Check for missing values.
5. Extract the passenger values.
6. Normalize the values using `MinMaxScaler`.
7. Convert the time series into supervised learning sequences.
8. Create training and testing sequences for each look-back window.

---

## 🏗️ Model Training

The project uses TensorFlow/Keras to construct and train the LSTM model.

The model contains:

* LSTM layer(s)
* Dropout layer(s)
* Dense output layer

`EarlyStopping` is used during training to help prevent unnecessary training when validation performance stops improving.

A fixed random seed of **42** is used to improve reproducibility.

---

## 📏 Evaluation Metrics

Two metrics are used to evaluate forecasting performance.

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

```text
MAE = Average(|Actual - Predicted|)
```

Lower MAE indicates smaller average prediction errors.

### RMSE — Root Mean Squared Error

Measures the square root of the average squared prediction error.

```text
RMSE = √(Average((Actual - Predicted)²))
```

Lower RMSE indicates better prediction accuracy.

---

## 📈 Model Comparison

The experiment compares the performance of:

| Look-Back Window | Historical Information |
| ---------------- | ---------------------- |
| 6 months         | Short-term pattern     |
| 12 months        | One seasonal cycle     |
| 24 months        | Two seasonal cycles    |

The final comparison is based on the **actual MAE and RMSE values produced by the notebook**.

The notebook also generates plots comparing:

* Actual passenger values
* Predicted passenger values
* Predictions from different look-back windows

---

## 📁 Project Structure

```text
.
├── Assignment6.ipynb
├── data/
│   └── airline-passengers.csv
└── README.md
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
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 3. Open the notebook

```text
Assignment6.ipynb
```

### 4. Run all cells

The notebook will:

* Download the dataset if required
* Preprocess the data
* Train the LSTM models
* Generate predictions
* Calculate MAE and RMSE
* Display comparison plots

---

## 🔑 Key Concepts

* Time-Series Forecasting
* LSTM Networks
* Sequential Data
* Look-Back Windows
* Time-Series Preprocessing
* Min-Max Scaling
* Supervised Learning from Time Series
* MAE
* RMSE
* Seasonal Patterns
* Model Comparison

---

## 📌 Conclusion

This project demonstrates the use of **LSTM neural networks for time-series forecasting** using monthly airline passenger data.

By comparing **6-month, 12-month, and 24-month look-back windows**, the experiment investigates how the amount of historical information provided to the LSTM affects forecasting performance.

The final model comparison is performed using **MAE and RMSE**, along with visual comparison of actual and predicted passenger values.
