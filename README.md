<h1>Gender and Age Prediction using Convolutional Neural Networks (CNN)</h1>

<h3>Overview</h3>
This repository contains code for a Convolutional Neural Network (CNN) model trained to predict gender and age from facial images. The model architecture is designed to accept input images and output two predictions simultaneously: gender (binary classification) and age (regression).

<h3>Model Architecture</h3>
The CNN model architecture consists of several layers:

<ul>
<li>Input layer: Accepts input images of specified dimensions.</li>
<li>Convolutional layers: Apply convolutional filters to extract features from the input images.</li>
<li>MaxPooling layers: Perform max pooling to downsample feature maps.</li>
<li>Flatten layer: Flatten the output from the convolutional layers to prepare for the fully connected layers.</li>
<li>Fully connected layers: Dense layers that learn complex patterns from the extracted features.</li>
<li>Dropout layers: Apply dropout regularization to prevent overfitting.</li>
<li>Output layers: Two output layers for gender (sigmoid activation) and age (relu activation) predictions.</li>
</ul>
  
<h3>Training</h3>
The model is trained using binary cross-entropy loss for gender prediction and mean absolute error (MAE) loss for age prediction. Additionally, accuracy is used as a metric for gender prediction, and MAE is used as a metric for age prediction. The model is optimized using the Adam optimizer.

<h3>Dependencies</h3>
The code is implemented using Python and requires the following dependencies, listed in requirements.txt:
<ul>
<li>TensorFlow</li>
<li>Keras</li>
<li>NumPy</li>
<li>Matplotlib (for visualization)</li>
<li>Seaborn (for visualization)</li>
<li>tqdm (for progress bars)</li></ul>

<h3>Dataset</h3>
The model is trained on a dataset containing facial images labeled with gender and age information. The dataset used for training is not included in this repository due to size constraints. It was taken from WWW.Kaggle.com and it contains over 23000 images with labels. (Ethnicity was ignored from image label for this project)

<ul><li>Dataset link: https://www.kaggle.com/datasets/jangedoo/utkface-new?select=crop_part1</li></ul>

<h3>Usage</h3>
To use the trained model for prediction:

<li>Load the saved model weights.</li>
<li>Preprocess input images (resize, grayscale conversion, etc.).</li>
<li>Feed the preprocessed images into the model for prediction.</li>

