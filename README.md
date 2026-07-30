# Breast Cancer Detection using Machine Learning

## Project Description
This project predicts whether a breast tumor is *Benign (Non-Cancerous)* or *Malignant (Cancerous)* using Machine Learning. The model is built using *Logistic Regression* and the *Breast Cancer Wisconsin Diagnostic Dataset*.


## Features
- Data Cleaning
- Feature Selection
- Label Encoding
- Train-Test Split
- Logistic Regression Model
- Model Prediction
- Accuracy Evaluation
- Confusion Matrix
- Classification Report

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Dataset Information
- Dataset Name: Breast Cancer Wisconsin Diagnostic Dataset
- File Format: CSV
- Total Records: 569
- Target Variable: Diagnosis
  - Benign (B) = 0
  - Malignant (M) = 1

## Features Used
- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean
- compactness_mean
- concavity_mean


## Machine Learning Algorithm
- Logistic Regression


## Project Workflow

1. Load Dataset
2. Clean the Data
3. Remove Unnecessary Columns
4. Select Important Features
5. Encode Target Labels
6. Split Dataset into Training and Testing Sets
7. Train Logistic Regression Model
8. Predict Test Data
9. Evaluate Model Performance


## Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report


## Python Libraries

python
pandas
scikit-learn


Install using:

bash
pip install pandas scikit-learn


## How to Run

1. Download the project.
2. Place the dataset (Cancer Detection dataset.csv) in the project folder.
3. Open the project in VS Code.
4. Run:

bash
python breast_cancer_detection.py

## Sample Output


Accuracy: 0.97

Confusion Matrix:
[[70  2]
 [ 1 41]]

Classification Report:

              precision    recall  f1-score   support

Benign          0.99       0.97      0.98       72
Malignant       0.95       0.98      0.96       42

accuracy                             0.97      114
macro avg        0.97       0.97      0.97      114
weighted avg     0.97       0.97      0.97      114

## Future Improvements
- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree
- Deep Learning (Neural Networks)
- Web-based Prediction System using Flask or Django
