# FYS-STK4155
This repository contains projects completed for the course FYS-STK4155 (Applied data analysis and machine learning) at the university of Oslo. Collaborators on the projects are Martine Tan, Aram Salihi, and Andras Philip Playan.

## List of projects

* **Project 3:** In this project, we compared Convolutional Neural Networks (CNN) with ordinary Logistic Regression for classifying images. We used two data sets: the famous MNIST data containing images of hand-written digits, and the following cats and dogs dataset from kaggle: https://www.kaggle.com/c/dogs-vs-cats/data. 
Unsurprisingly, CNNs drastically outperformed logistic regression.
We used Tensorflow to build CNNs in this project.
Excerpt from the [project report](https://github.com/martinekbh/FYS-STK4155/blob/master/project3/report_for_project_3.pdf):
  >In this numerical study we are going to compare two machine learning methods for image classification: Logistic Regression and Convolutional Neural Networks (CNN). We will use two image data sets. The first data set is the well known MNIST data with hand-written digits, consisting of 60000 images with a resolution of (28 x 28) pixels. The other data set consists of 5000 images of cats and dogs, with a chosen resolution of (150 x 150 x 3) pixels. The goal of this study is to show that when the number of pixels and image complexity (background noise, unfocused objects, varying contrast levels, etc.) increases, the accuracy a simple machine learning method such as logistic regression is able to achieve decreases drastically, and that CNNs are much more suited to such tasks. On the MNIST data set, logistic regression was able to achieve an accuracy score of 92.56%, while our best CNN model achieved 98.76% accuracy. On the cats and dogs data, logistic regression barely outperformed random classification, with an accuracy of 53.07%, while our best CNN model achieved an 81.65% accuracy score. We conclude that CNN tremendously outperforms logistic regression in image classification, and that logistic regression is unsuitable for complex image data.

* **Project 2:** In this project we developed our own implementation of Logistic Regression and Feed-Forward Neural Networks (FFNN), and used these to analyze the Wisconsin breat cancer data set (https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)). The goal was to compare the performance of logistic regression and FFNNs in analyzing the data. We also trained a FFNN on data generated by Franke's function, and compared its performance with the performance of the regression models tested on the same data in the previous project.
Excerpt from the abstract of the [project report](https://github.com/martinekbh/FYS-STK4155/blob/master/project2/Report/FYS-STK4155_project_2.pdf):
  >In this numerical study we have developed our own implementation of logistic regression and feed-forward neural network in Python, and use these implementations to analyze the Wisconsin breast cancer dataset, as well as the Franke's function dataset. The latter dataset is one we have generated using Franke's function. It consists of 10,000 observations, to which we have added noise drawn from the Gaussian distribution with mean 0.001 and standard deviation 0.1. We concluded that the feed-forward neural network outperforms logistic regression at classification. The feed-forward neural network was able to predict whether patients had breast cancer or not with an 99.1% accuracy score. With logistic regression, the accuracy score was 96.5%. The feed-forward Neural Network was able to estimate Franke's function with a mean squared error of 0.01002, and R2-score of 0.89386. This score is very close to the score we obtain with ridge regression: a mean squared error of 0.00999 and an R2-score of 0.89165. However, the neural network algorithm has potential for further improvement, and performance may be improved by fine tuning the network parameters. Both the logistic regression and feed-forward neural network implementation use stochastic gradient descent as optimization method.

* **Project 1:** In this project we implemented algorithms for Linear Regression, Ridge Regression and Lasso (using python), and compared how well these regression methods were able to model Franke's function and digital terrain data from https://earthexplorer.usgs.gov/.
Excerpt from the introduction of the [project report](https://github.com/martinekbh/FYS-STK4155/blob/master/project1/article/fys_stk_project1.pdf): 
  >The aim of this numerical study is to experiment with three different regression methods in detail: Linear regression, ridge regression, and lasso regression. In addition, cross validation will be used as a resampling technique to further evaluate the regression methods. The project largely consists of two parts. The first part comprises of implementing the regression methods and relevant algorithms in python. We will use Franke's function to generate a data set upon which we can apply and test our code during the developing process. We will also discuss in detail the bias-variance trade-off in the regression methods. In the second part of the our numerical study, we will apply this code to digital terrain data from USGS. Digital terrain data can be downloaded from https://earthexplorer.usgs.gov/. 
