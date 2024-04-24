Gender and Age Prediction using Convolutional Neural Networks (CNN)
Overview
This repository contains code for a Convolutional Neural Network (CNN) model trained to predict gender and age from facial images. The model architecture is designed to accept input images and output two predictions simultaneously: gender (binary classification) and age (regression).

Model Architecture
The CNN model architecture consists of several layers:

Input layer: Accepts input images of specified dimensions.
Convolutional layers: Apply convolutional filters to extract features from the input images.
MaxPooling layers: Perform max pooling to downsample feature maps.
Flatten layer: Flatten the output from the convolutional layers to prepare for the fully connected layers.
Fully connected layers: Dense layers that learn complex patterns from the extracted features.
Dropout layers: Apply dropout regularization to prevent overfitting.
Output layers: Two output layers for gender (sigmoid activation) and age (relu activation) predictions.
Training
The model is trained using binary cross-entropy loss for gender prediction and mean absolute error (MAE) loss for age prediction. Additionally, accuracy is used as a metric for gender prediction, and MAE is used as a metric for age prediction. The model is optimized using the Adam optimizer.

Dependencies
The code is implemented using Python and requires the following dependencies, listed in requirements.txt:

TensorFlow (>=2.0)
Keras
NumPy
Matplotlib (for visualization)
Seaborn (for visualization)
tqdm (for progress bars)

Dataset
The model is trained on a dataset containing facial images labeled with gender and age information. The dataset used for training is not included in this repository due to size constraints. It was taken from WWW.Kaggle.com and it contains over 23000 images with labels. (Ethnicity was ignored from image label for this project)
Dataset link: https://www.kaggle.com/datasets/jangedoo/utkface-new?select=crop_part1

Usage
To use the trained model for prediction:

Load the saved model weights.
Preprocess input images (resize, grayscale conversion, etc.).
Feed the preprocessed images into the model for prediction.

