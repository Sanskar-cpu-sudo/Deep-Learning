# Deep Learning Audio Classification with Speech-to-Text and Text-to-Speech

## 📌 Project Overview

This project implements a deep learning-based audio classification system using a **Convolutional Neural Network (CNN)** and TensorFlow. It classifies short spoken commands from audio recordings using spectrogram-based feature extraction.

The project also integrates **Speech-to-Text (STT)** to transcribe spoken audio and **Text-to-Speech (TTS)** to generate spoken responses, creating a complete audio-processing pipeline.

The system is built using Python and demonstrates audio preprocessing, deep learning model training, performance evaluation, speech recognition, and speech synthesis.

## 🎯 Objectives

- Build a CNN model to classify spoken commands.
- Process raw audio signals and convert them into spectrograms.
- Train and evaluate the model using separate training, validation, and testing datasets.
- Measure performance using accuracy, precision, recall, and F1-score.
- Implement speech-to-text transcription.
- Generate spoken responses using text-to-speech.
- Save the trained model and evaluation outputs for future use.

## ✨ Features

- **Audio Classification:** Predicts the class of an audio recording using a trained CNN.
- **Audio Preprocessing:** Resamples audio to 16 kHz, converts it to mono, and pads or truncates it to a fixed duration.
- **Spectrogram Generation:** Converts audio waveforms into log-magnitude spectrograms.
- **CNN Architecture:** Uses convolutional layers, batch normalization, max pooling, and dropout.
- **Model Training:** Supports validation, early stopping, and best-model checkpointing.
- **Performance Evaluation:** Generates classification metrics and a confusion matrix.
- **Speech-to-Text:** Transcribes audio using the Google Speech Recognition service.
- **Text-to-Speech:** Generates spoken output using Google Text-to-Speech (gTTS).
- **Model Persistence:** Saves the trained model in Keras format.
- **Visualizations:** Plots spectrograms, training curves, and classification results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN model development and training |
| Librosa | Audio loading and preprocessing |
| NumPy | Numerical computations |
| Pandas | Dataset organization and evaluation reports |
| Matplotlib | Visualization of spectrograms and training curves |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Model evaluation metrics |
| SpeechRecognition | Speech-to-text transcription |
| Google Text-to-Speech (gTTS) | Text-to-speech generation |
| IPython | Audio playback inside the notebook |
| Google Colab | Notebook execution and file uploads |

## 📂 Dataset

The project uses the **Mini Speech Commands dataset** provided by TensorFlow.

Dataset URL: https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip

The dataset contains short audio recordings organized into folders according to their spoken-command labels.

The notebook automatically downloads and extracts the dataset, validates WAV files, identifies available classes, and assigns numerical labels to them.

### Dataset Split

The dataset is divided into three subsets:

- **Training set — 70%:** Used to train the CNN.
- **Validation set — 15%:** Used to monitor performance and select the best model.
- **Testing set — 15%:** Used for final performance evaluation.

The split is performed separately for each class using a fixed random seed for reproducibility.

## 🧠 Model Architecture

The system uses a CNN to learn patterns from log-magnitude audio spectrograms.

### Audio Processing Pipeline

1. Load the audio recording.
2. Resample it to 16,000 Hz and convert it to mono.
3. Pad or truncate the waveform to one second.
4. Compute the Short-Time Fourier Transform (STFT).
5. Convert the magnitude spectrum into a logarithmic representation.
6. Pass the spectrogram to the CNN.
7. Generate class probabilities using the softmax output layer.
8. Select the class with the highest predicted probability.

### CNN Architecture

| Layer | Configuration |
|---|---|
| Input | Log-magnitude spectrogram |
| Convolutional Layer 1 | 32 filters, 3 × 3 kernel, ReLU |
| Batch Normalization | Normalizes intermediate activations |
| Max Pooling | Reduces spatial dimensions |
| Convolutional Layer 2 | 64 filters, 3 × 3 kernel, ReLU |
| Batch Normalization | Normalizes intermediate activations |
| Max Pooling | Reduces spatial dimensions |
| Convolutional Layer 3 | 128 filters, 3 × 3 kernel, ReLU |
| Batch Normalization | Normalizes intermediate activations |
| Global Average Pooling | Aggregates spatial features |
| Dropout | 35% dropout rate |
| Output Layer | Softmax classification over available classes |

The model is trained using the Adam optimizer with a learning rate of `0.001` and sparse categorical cross-entropy loss.

Training runs for up to 20 epochs, with early stopping based on validation loss.

## 🔄 Speech-to-Text and Text-to-Speech

### Speech-to-Text (STT)

The SpeechRecognition library processes an audio recording and sends it to the Google Speech Recognition service for transcription.

The function returns the recognized text or an empty string when transcription fails.

### Text-to-Speech (TTS)

The gTTS library converts text into speech and saves the generated audio as an MP3 file.

In the demonstration, the system speaks a response based on the CNN's predicted command.

**Note:** STT and TTS require an internet connection. They use external services and are separate from the CNN's audio-classification process.

## 📊 Evaluation Metrics

The model is evaluated on the held-out test dataset using the following metrics:

- **Accuracy:** Proportion of correctly classified test samples.
- **Precision:** Measures how often predictions for a class are correct.
- **Recall:** Measures how many actual samples of a class are identified correctly.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

Macro-averaged precision, recall, and F1-score are reported to give equal weight to each class.

The notebook also records the model's training time and saves the evaluation metrics to a CSV file.

*Actual evaluation scores depend on the training run and are generated when the notebook executes.*

## ⚙️ Installation and Setup

### Option 1: Run on Google Colab (Recommended)

1. Download the project notebook.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select **File → Upload notebook**.
4. Upload `audio_classification_stt_tts_notebook.ipynb`.
5. Run the cells sequentially from top to bottom.
6. Wait for the dataset download and CNN training to complete.
7. Upload a WAV audio file when prompted to test the system.

The notebook installs its required Python packages automatically.

### Option 2: Run Locally

Create a Python virtual environment and install the required dependencies:

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn librosa scikit-learn SpeechRecognition gTTS ipython
```

Open the notebook using Jupyter:

```bash
pip install notebook
jupyter notebook
```

**Important:** The provided notebook uses Google Colab-specific paths and `google.colab.files.upload()`. For local execution, replace those parts with local file paths or an appropriate file-upload mechanism.

## 🚀 How to Use

After training is complete:

1. Upload a WAV audio recording.
2. The preprocessing pipeline converts the recording into a spectrogram.
3. The CNN predicts the most likely audio class.
4. The system displays the predicted class and model confidence.
5. The STT module attempts to transcribe the recording.
6. The TTS module generates a spoken response describing the predicted command.
7. The notebook displays a final summary containing the transcription, predicted class, and confidence.

For best results, use a clear recording containing a command represented in the training dataset. The classifier is designed for one-second audio clips, so longer recordings are truncated and shorter recordings are padded.

## 📁 Project Outputs

The notebook generates the following files in Google Colab:

| File | Description |
|---|---|
| `audio_classification_model.keras` | Saved trained CNN model |
| `best_audio_classifier.keras` | Best model checkpoint selected using validation loss |
| `evaluation_results.csv` | Evaluation metrics |
| `training_curves.png` | Training and validation accuracy/loss plots |
| `confusion_matrix.png` | Confusion matrix visualization |
| `speech_output.mp3` | Generated spoken response, if TTS succeeds |

These files are saved under `/content/` when generated successfully.

## 🔮 Future Enhancements

- Develop a real-time microphone-based audio classification interface.
- Build a web application for uploading and classifying audio files.
- Add noise reduction and audio augmentation to improve robustness.
- Explore advanced architectures such as CRNNs and transformer-based audio models.
- Integrate an offline speech recognition engine.
- Add multilingual speech recognition and speech synthesis.
- Optimize the model for deployment on mobile or edge devices.
- Evaluate the model under different noise conditions and recording environments.

## ⚠️ Limitations

- The classifier recognizes only the classes present in its training dataset.
- The model is designed for short, one-second audio inputs.
- Predictions may be less reliable for noisy recordings, unfamiliar speakers, or out-of-distribution audio.
- Speech recognition and speech synthesis require internet access.
- Model confidence is a predicted probability and should not automatically be interpreted as a calibrated measure of certainty.
- Training time and evaluation scores depend on the execution environment and training run.

## 👨‍💻 Applications

Potential applications include:

- Voice-controlled interfaces.
- Basic voice-command recognition.
- Educational demonstrations of deep learning and audio processing.
- Audio-based human-computer interaction.
- Prototyping speech-enabled applications.

## 📜 License

This project is intended for educational and research purposes. If you distribute the project, specify an appropriate license for your own code and comply with the licenses and terms of the third-party libraries, dataset, and external speech services.

## 🙌 Acknowledgements

- TensorFlow for the Mini Speech Commands dataset and deep learning framework.
- The Librosa community for audio analysis tools.
- The Scikit-learn community for evaluation utilities.
- The SpeechRecognition and gTTS projects for speech processing integrations.

---

**Project:** Deep Learning Audio Classification with STT and TTS  
**Core model:** Convolutional Neural Network (CNN)  
**Framework:** TensorFlow / Keras  
**Execution environment:** Google Colab / Jupyter Notebook

