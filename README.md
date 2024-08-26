# Facial Emotion Recognition (FER) with CNN

This project implements a Convolutional Neural Network (CNN) to classify facial emotions using the FER2013 dataset. The model categorizes images into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Overview

- **Dataset**: FER2013, containing 48x48 grayscale images of faces.
- **Model**: A deep CNN with five convolutional blocks, batch normalization, max pooling, and dropout layers.
- **Output**: Trained model capable of predicting emotions from facial images.

## Quick Start

1. **Mount Google Drive** (if using Google Colab):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Load Dataset**:
    ```python
    df = pd.read_csv('/content/drive/My Drive/fer.csv')
    ```

3. **Train Model**:
    ```python
    history = model.fit_generator(train_flow, epochs=100, validation_data=test_flow)
    ```

4. **Evaluate Model**:
    ```python
    loss = model.evaluate(X_test/255., y_test)
    print("Test Loss:", loss[0])
    print("Test Accuracy:", loss[1])
    ```

5. **Save Model**:
    ```python
    model.save('/content/drive/My Drive/Colab Notebooks/Fer2013.h5')
    ```

## Requirements

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

Install dependencies with:
```bash
pip install tensorflow keras numpy pandas matplotlib
