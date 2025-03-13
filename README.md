# learning-algorithms

This repository contains implementations of machine learning algorithms for three different problems:

1. Handwriting Recognition using Classifiers with MNIST Data
    * Logistic Regression Accuracy: 0.9050
    * SVM Accuracy: 0.9170
    * Random Forest Accuracy: 0.9378

2. Weather Prediction using Supervised Learning
    * Best Model: Random Forest (Accuracy: 83%)
    * Other Models: SVM (78%), Linear Regression (35%)

3. Penguin Species Classification using K-Means Clustering
    * K-Means Accuracy: 77%

## Installation Instructions

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Handwriting Recognition

* Dataset: MNIST
* Algorithms Used:
    * Logistic Regression → 90.50% accuracy
    * SVM → 91.70% accuracy
    * Random Forest → 93.78% accuracy
* Evaluation: Accuracy, Mean Squared Error (MSE), Visualization

## Weather Prediction

* Dataset: seattle-weather.csv
* Algorithms Used:
    * Random Forest → Best performance (83% accuracy, MSE: 1.35)
    * SVM → 78% accuracy, MSE: 1.60
    * Linear Regression → 35% accuracy, MSE: 1.16
* Evaluation: Accuracy, MSE, Confusion Matrix

## Penguin Species Classification

* Dataset: penguins.csv
* Algorithm Used: K-Means Clustering (K=3)
* Accuracy: 77% (compared to actual labels)
