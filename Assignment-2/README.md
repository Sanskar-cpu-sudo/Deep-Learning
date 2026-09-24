# Assignment 2 – Multilayer Perceptron (MLP) for Iris Classification

## Aim

To design and implement a **Multilayer Perceptron (MLP)** neural network for classification of the **Iris dataset** and evaluate its performance using **accuracy** and a **confusion matrix**.

## Dataset

The **Iris dataset** from Scikit-learn is used.

* **Total samples:** 150
* **Features:** 4
* **Classes:** 3

  * Setosa
  * Versicolor
  * Virginica

### Features

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

## Technologies Used

* Python
* NumPy
* Matplotlib
* Scikit-learn
* MLPClassifier

## Methodology

The following steps were performed:

1. Loaded the Iris dataset.
2. Split the dataset into training and testing sets.
3. Used **80% data for training** and **20% for testing**.
4. Applied **StandardScaler** for feature scaling.
5. Created an MLP classifier with:

   * Two hidden layers
   * 10 neurons in each hidden layer
   * ReLU activation function
   * Adam optimizer
6. Trained the MLP model using the training data.
7. Predicted the classes for the test data.
8. Calculated the classification accuracy.
9. Generated and displayed the confusion matrix.

## MLP Architecture

```text
Input Layer
    ↓
4 Input Features
    ↓
Hidden Layer 1
10 Neurons + ReLU
    ↓
Hidden Layer 2
10 Neurons + ReLU
    ↓
Output Layer
3 Classes
```

## Model Configuration

```python
MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
```

## Results

### Accuracy

The model achieved:

**Accuracy = 96.67%**

### Confusion Matrix

```text
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

The confusion matrix shows that:

* **10 Setosa** samples were correctly classified.
* **9 Versicolor** samples were correctly classified and **1 was classified as Virginica**.
* **10 Virginica** samples were correctly classified.

## Conclusion

The MLP classifier successfully classified the Iris dataset with an accuracy of **96.67%**. The confusion matrix shows that the model correctly classified most of the test samples, with only one misclassification.

## Files

```text
Assignment-1/
│
├── lab2.ipynb
└── README.md
```

