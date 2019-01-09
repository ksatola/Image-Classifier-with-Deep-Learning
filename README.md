# Developing an Image Classifier with Deep Learning

## Overview
In this project, I will train an image classifier to recognize different species of flowers, then export it for use in a stand alone application.

This project was completed as part of Udacity's [Data Scientist Nanodegree](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) certification.

## Objectives
1. Implement an image classifier with PyTorch (build and train a neural network on the flower data set).
2. Convert the classifier into an application that others can use. The application should be a pair of Python scripts that run from the command line. The first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint. The second file, `predict.py`, uses a trained network to predict the class for an input image. 

## Data Origin
The [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) consisting of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images. The details of the categories and the number of images for each class can be found on this [category statistics page](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html).

The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features.


## Tools, Software and Libraries
This project uses the following software and Python libraries:
- Python 3.6
- torch, torchvision
- numpy
- json
- time
- os
- random
- matplotlib
- seaborn
- PIL
- argparse

## Results
A Python application that can be trained with PyTorch on any set of labeled images.
The project is broken down into multiple steps:
- Load and preprocess the image dataset,
- Train the image classifier on your dataset,
- Use the trained classifier to predict image content.

## Details
- [HTML Preview](https://ksatola.github.io/projects/image_classifier_with_deep_learning.html)
- [Jupyter Notebook](https://github.com/ksatola/Image-Classifier-with-Deep-Learning/blob/master/final4.ipynb)

### train.py

![train1.png](/assets/train1.png)
![train2.png](/assets/train2.png)
![train3.png](/assets/train3.png)
![train4.png](/assets/train4.png)
![train5.png](/assets/train5.png)

### predict.py
![predict1.png](/assets/predict1.png)
![predict2.png](/assets/predict2.png)
