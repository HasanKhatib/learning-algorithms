# learning-algorithms

This repository contains implementations of machine learning algorithms for two different problems:

1.	Weather Prediction using Supervised Learning
    * Best Model: Random Forest (Accuracy: 83%)
	* Other Models: SVM (78%), Linear Regression (35%)
2.	Penguin Species Classification using K-Means Clustering
	* K-Means Accuracy: 77%

## Installation Instructions

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

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
