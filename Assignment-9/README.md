# Vision Transformer vs. CNN for CIFAR-10 Image Classification

## Overview

This project compares a pretrained **Vision Transformer (ViT)** with a pretrained **ResNet-18 Convolutional Neural Network (CNN)** for image classification using the CIFAR-10 dataset. Both models use transfer learning, with pretrained feature-extraction layers frozen and classification heads trained to classify images into 10 categories.

The project evaluates model performance using accuracy, precision, recall, F1-score, inference time, and confusion matrices.

## Models Used

### 1. Vision Transformer (ViT)
- Pretrained model: `google/vit-base-patch16-224`
- Loaded using Hugging Face Transformers.
- The pretrained encoder is frozen.
- The classification head is trained for 10 classes.

### 2. ResNet-18 CNN
- Pretrained ResNet-18 model from Torchvision.
- The convolutional feature extractor is frozen.
- The final fully connected layer is replaced with a 10-class output layer.
- Only the final classification layer is trained.

**Note:** This experiment uses frozen-backbone transfer learning rather than full model fine-tuning.

## Dataset

The project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which contains images belonging to 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

To reduce training time, the notebook uses the following subsets:

| Dataset Split | Number of Images |
|---|---:|
| Training | 10,000 |
| Testing | 2,000 |

### Data Preprocessing

- Resize images to 224 × 224 pixels.
- Convert images into PyTorch tensors.
- Normalize images using ImageNet mean and standard deviation.
- Load data using PyTorch DataLoaders.

## Project Workflow

1. Import the required libraries.
2. Set random seeds for reproducibility.
3. Select CUDA or CPU as the computing device.
4. Download and prepare the CIFAR-10 dataset.
5. Create training and testing subsets.
6. Visualize sample images.
7. Load pretrained ViT and ResNet-18 models.
8. Freeze pretrained feature extractors.
9. Train the classification heads for three epochs.
10. Evaluate both models on the test dataset.
11. Compare evaluation metrics and confusion matrices.
12. Generate predictions for an example image.

## Evaluation Metrics

The notebook evaluates both models using the following metrics:

- **Accuracy:** Percentage of correctly classified images.
- **Precision:** Measures how many predicted positive classifications are correct.
- **Recall:** Measures how many actual class instances are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Inference Time:** Average time taken to process each test image.

Macro-averaged precision, recall, and F1-score are used to evaluate performance across all classes equally.

The notebook also generates:
- Classification reports
- Accuracy comparison bar chart
- Confusion matrices
- Example image predictions

Actual performance metrics are produced when the notebook is executed.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Hugging Face Transformers
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

## Installation

Install the required dependencies:

```bash
pip install torch torchvision transformers scikit-learn seaborn matplotlib numpy pandas jupyter
```

## How to Run

### Google Colab

1. Upload `Assignment_9.ipynb` to Google Colab.
2. Execute the notebook cells in order.
3. Allow the dataset and pretrained model weights to download.
4. Review the training results, evaluation metrics, graphs, and predictions.

### Local Environment

1. Clone or download the repository.
2. Install the required dependencies.
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `Assignment_9.ipynb`.
5. Run all cells sequentially.

A CUDA-enabled GPU is recommended for faster execution. The notebook can also run on a CPU, although training and inference may take longer.

An internet connection is required for the initial download of the dataset and pretrained model weights.

## Training Configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Training samples | 10,000 |
| Testing samples | 2,000 |
| Batch size | 32 |
| Number of epochs | 3 |
| ViT optimizer | AdamW |
| ResNet-18 optimizer | Adam |
| Learning rate | 0.001 |
| Number of classes | 10 |
| Input image size | 224 × 224 |

## Repository Structure

```text
.
├── Assignment_9.ipynb
└── README.md
```

## Future Improvements

- Train using the complete CIFAR-10 dataset.
- Experiment with different learning rates and training epochs.
- Fine-tune selected layers or the entire pretrained models.
- Compare additional CNN and Vision Transformer architectures.
- Save trained model checkpoints for reuse.
- Repeat experiments with multiple random seeds.
- Benchmark inference performance on consistent hardware.

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)

