# Metal Defect Detection using Machine Learning and Deep Learning

This project focuses on detecting defects in metal images using both traditional machine learning techniques and deep learning with convolutional neural networks (CNNs). The goal is to classify metal images as either defective or non-defective.

## Introduction

The project consists of the following main steps:

1. Downloading the images.
2. Data augmentation and preprocessing.
3. Applying principal component analysis (PCA) for dimensionality reduction.
4. Training a random forest classifier on the original data.
5. Training another random forest classifier after applying PCA.
6. Implementing a CNN using Keras for defect detection.
7. Training the CNN on the dataset and evaluating its performance.
8. Predicting defect presence using both the random forest classifier and the CNN.

## Code Overview

- **Downloading and Preprocessing**: 
  - The images are downloaded and preprocessed, including data augmentation.
  - Images are converted to grayscale, resized to (32,32), and flattened for feature extraction.

- **Applying PCA**:
  - Principal Component Analysis (PCA) is applied to reduce the dimensionality of the data.

- **Random Forest Classifier**:
  - A random forest classifier is trained on both the original and PCA-transformed data.
  - Accuracy is evaluated on the test set.

- **Convolutional Neural Network (CNN)**:
  - A CNN model is defined using Keras for defect detection.
  - The model consists of convolutional layers followed by max-pooling layers and fully connected layers.
  - The model is trained and evaluated on the dataset.

- **Prediction**:
  - The trained models are used to predict whether a metal image contains defects.
  - Both the random forest classifier and the CNN are used for prediction.

## Requirements

The following libraries are required to run the code:
- numpy
- opencv-python
- tqdm
- matplotlib
- scikit-learn
- keras

You can install the required libraries using `pip install -r requirements.txt`.

## Usage

1. Make sure you have the required libraries installed.
2. Run the `metal_defect_detection.ipynb` notebook.

## References

- [Keras documentation](https://keras.io/)
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [OpenCV documentation](https://docs.opencv.org/)
