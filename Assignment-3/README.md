# Forward Propagation and Backpropagation using TensorFlow/Keras

## Aim

To implement **forward propagation and backpropagation** using TensorFlow/Keras and analyze the effect of different **learning rates** and **number of epochs** on model performance.

---

## Dataset

The **Wine Dataset** from Scikit-learn is used for this experiment.

The dataset contains chemical measurements of wines belonging to **3 different classes**.

### Dataset Information

* **Number of samples:** 178
* **Number of features:** 13
* **Number of classes:** 3
* **Dataset:** Wine Dataset
* **Source:** `sklearn.datasets.load_wine()`

The 13 features represent different chemical properties of wine, such as alcohol, malic acid, magnesium, flavanoids, and others.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Scikit-learn
* Matplotlib

---

## Libraries Used

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
```

---

## Methodology

The following steps are performed in the experiment:

1. Load the Wine dataset.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets.
4. Standardize the input features.
5. Create a neural network using Keras.
6. Perform forward propagation through the network.
7. Calculate the loss using categorical cross-entropy.
8. Perform backpropagation to calculate gradients.
9. Update the weights using the Adam optimizer.
10. Train the model for different numbers of epochs.
11. Repeat the experiment using different learning rates.
12. Evaluate the model using test accuracy.
13. Compare the performance of different combinations.

---

## Neural Network Architecture

The neural network consists of the following layers:

| Layer          | Neurons | Activation |
| -------------- | ------: | ---------- |
| Input Layer    |      13 | -          |
| Hidden Layer 1 |      16 | ReLU       |
| Hidden Layer 2 |      12 | ReLU       |
| Output Layer   |       3 | Softmax    |

### Architecture

```text
Wine Dataset
     |
     | 13 Features
     ↓
Input Layer
     |
     ↓
Dense Layer
16 Neurons
ReLU
     |
     ↓
Dense Layer
12 Neurons
ReLU
     |
     ↓
Output Layer
3 Neurons
Softmax
     |
     ↓
Predicted Class
```

---

## Forward Propagation

During forward propagation, the input data passes through each layer of the neural network.

For each neuron, the weighted sum is calculated:

```text
Z = WX + b
```

The activation function is then applied.

For the hidden layers, **ReLU** is used:

```text
ReLU(x) = max(0, x)
```

The output layer uses **Softmax** to produce probabilities for the three wine classes.

---

## Backpropagation

Backpropagation is used to update the weights of the neural network.

The process is:

1. Calculate the prediction using forward propagation.
2. Calculate the loss between predicted and actual values.
3. Calculate gradients of the loss with respect to the weights.
4. Update the weights using the optimizer.
5. Repeat the process for the specified number of epochs.

The Adam optimizer is used for updating the weights.

---

## Learning Rate

The learning rate controls how much the model's weights are changed during each update.

The following learning rates are tested:

```python
learning_rates = [0.0005, 0.001, 0.01]
```

A very small learning rate may make training slow, while a very large learning rate can cause unstable learning.

---

## Number of Epochs

An epoch represents one complete pass through the training dataset.

The following epoch values are tested:

```python
epochs_list = [5, 7, 9]
```

Each learning rate is tested with each epoch value.

Therefore:

```text
3 Learning Rates × 3 Epoch Values = 9 Experiments
```

The experiments are:

| Learning Rate | Epochs |
| ------------: | -----: |
|        0.0005 |      5 |
|        0.0005 |      7 |
|        0.0005 |      9 |
|         0.001 |      5 |
|         0.001 |      7 |
|         0.001 |      9 |
|          0.01 |      5 |
|          0.01 |      7 |
|          0.01 |      9 |

---

## Experimental Results

The model produces a performance summary similar to:

```text
Performance Summary
----------------------------------------------
Learning Rate    Epochs    Accuracy
----------------------------------------------
0.0005           5         0.6000
0.0005           7         0.3333
0.0005           9         0.6889
0.001            5         0.4444
0.001            7         0.8444
0.001            9         0.6667
0.01             5         0.9778
0.01             7         0.9778
0.01             9         1.0000
```

> Note: The exact accuracy values can change between runs because neural networks use randomly initialized weights.

---

## Result Analysis

The experiment shows that the **learning rate has a significant effect on model performance**.

For the tested combinations, a learning rate of `0.01` produced higher test accuracy than `0.0005` and `0.001`.

Increasing the number of epochs does not always guarantee higher test accuracy. The effect depends on the learning rate and the training process.

For example, in the observed results:

* `0.0005` with 5 epochs → **60.00%**
* `0.001` with 7 epochs → **84.44%**
* `0.01` with 5 epochs → **97.78%**
* `0.01` with 9 epochs → **100.00%**

Thus, both the **learning rate** and **number of epochs** influence the final performance of the neural network.

---

## Accuracy Graph

The experiment also plots training and validation accuracy.

```python
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Wine Dataset Accuracy")
plt.legend()
plt.show()
```

The graph helps observe how the model's accuracy changes as training progresses.

---

## Algorithm

1. Start.
2. Load the Wine dataset.
3. Separate features and target labels.
4. Split the data into training and testing sets.
5. Standardize the features.
6. Create the neural network.
7. Initialize the weights and biases.
8. Perform forward propagation.
9. Calculate the loss.
10. Perform backpropagation.
11. Update weights using the Adam optimizer.
12. Repeat the process for the specified number of epochs.
13. Test different learning rates.
14. Test different epoch values.
15. Calculate test accuracy for each combination.
16. Compare the obtained results.
17. Plot the training and validation accuracy.
18. Stop.

---

## Conclusion

The experiment successfully implements **forward propagation and backpropagation using TensorFlow/Keras** on the Wine dataset.

The results demonstrate that the choice of **learning rate** and **number of epochs** affects neural network performance. A suitable learning rate allows the model to converge effectively, while an appropriate number of epochs provides sufficient training without unnecessary computation or potential overfitting.

The experiment also demonstrates how different hyperparameter combinations can be compared to understand their effect on model accuracy.

---

## How to Run

### 1. Install required libraries

```bash
pip install tensorflow scikit-learn numpy matplotlib
```

### 2. Run the Python program

```bash
python main.py
```

### 3. View the output

The program displays:

* Test accuracy for each experiment
* Performance summary
* Training vs validation accuracy graph

---

## Project Structure

```text
Forward-Backpropagation/
│
├── lab3.py
└── README.md
```

---

## Key Learning Outcomes

After completing this experiment, the following concepts are understood:

* Neural network architecture
* Forward propagation
* Backpropagation
* Activation functions
* Loss functions
* Adam optimizer
* Learning rate
* Epochs
* Model training
* Model evaluation
* Hyperparameter analysis
* Training and validation accuracy
