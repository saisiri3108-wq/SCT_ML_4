# Hand Gesture Recognition using CNN

## Overview

This project implements a Hand Gesture Recognition System using a Convolutional Neural Network (CNN) and the American Sign Language (ASL) Alphabet Dataset. The model is trained to recognize hand gestures representing different ASL alphabets.

## Features

* Image preprocessing and normalization
* Data augmentation for improved model performance
* CNN-based image classification
* Training and validation accuracy visualization
* Prediction on custom hand gesture images
* Model saving and loading

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

## Dataset

ASL Alphabet Dataset containing images of:

* A to Z alphabets
* Space
* Delete (Del)
* Nothing

Dataset Structure:

asl_alphabet_train/
├── A/
├── B/
├── C/
├── ...
├── Z/
├── del/
├── nothing/
└── space/

## Model Architecture

* Conv2D Layer (32 filters)
* MaxPooling2D
* Conv2D Layer (64 filters)
* MaxPooling2D
* Conv2D Layer (128 filters)
* MaxPooling2D
* Flatten Layer
* Dense Layer (128 neurons)
* Dropout Layer
* Output Layer (Softmax)

## Installation

1. Clone the repository

git clone https://github.com/saisiri3108-wq/SCT_ML_4.git

2. Install required libraries

pip install tensorflow matplotlib numpy

3. Run the training script

python hand.py

## Output

The model:

* Trains on the ASL dataset
* Evaluates validation accuracy
* Saves the trained model
* Displays accuracy and loss graphs

## Files

* hand.py – Model training script
* predict.py – Predict hand gestures from images
* output_shape.py – Check model architecture
* README.md – Project documentation

## Learning Outcomes

* Deep Learning Fundamentals
* Convolutional Neural Networks (CNN)
* Image Classification
* Data Augmentation
* TensorFlow & Keras
* Computer Vision Concepts

## Author

Sai Siri

Machine Learning Internship – SkillCraft Technology
