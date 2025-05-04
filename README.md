# Diabetes Prediction

A machine learning project for predicting diabetes based on diagnostic measurements.

## Overview

This project uses machine learning to predict whether a patient has diabetes based on certain diagnostic measurements. The implementation utilizes a dataset from the National Institute of Diabetes and Digestive and Kidney Diseases and employs various classification algorithms to achieve accurate prediction results.

## Dataset

The dataset used in this project is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It consists of several medical predictor variables and one target variable, 'Outcome'. Predictor variables include:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skinfold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: A function that scores likelihood of diabetes based on family history
- Age: Age in years

The target variable 'Outcome' is binary (0 for non-diabetic, 1 for diabetic).

## Project Structure

├── data/
│   └── diabetes.csv            # The dataset used for training and testing
├── src/
│   ├── data_preprocessing.py   # Functions for data preprocessing and analysis
│   ├── model_training.py       # Functions for training various models
│   └── model_evaluation.py     # Functions for evaluating model performance
├── notebooks/
│   ├── data_exploration.ipynb  # Jupyter notebook for data exploration
│   └── model_comparison.ipynb  # Jupyter notebook comparing different models
├── models/
│   └── trained_models/         # Directory to store trained models
├── requirements.txt            # Required Python packages
└── main.py                     # Main execution file

## Features

- Data preprocessing and cleaning
- Exploratory data analysis with visualizations
- Implementation of multiple classification algorithms:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Hyperparameter tuning using Grid Search
- Model evaluation metrics including accuracy, precision, recall, and F1-score
- Cross-validation for robust model assessment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YuMe-02/DiabetesPrediction.git
cd DiabetesPrediction
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

Usage
Running the main script:

```python main.py
```

This will execute the complete pipeline: data preprocessing, model training, and evaluation.
Using the notebooks:
To explore the data and understand the model comparison process, you can run the Jupyter notebooks in the notebooks directory:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

Results
The project compares several machine learning models for diabetes prediction. Performance metrics are evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC

The final model achieves an accuracy of approximately 78% on the test set, with balanced precision and recall scores.

Acknowledgments

The original dataset is from the National Institute of Diabetes and Digestive and Kidney Diseases
Inspiration from various healthcare machine learning applications
