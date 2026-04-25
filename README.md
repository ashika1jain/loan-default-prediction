# Loan Default Prediction

A machine learning model to predict whether a LendingClub borrower will repay or default on their loan.

## Live Demo
[Click here to try the app](https://loan-default-prediction-lendingclub.streamlit.app)

## Problem Statement
Standard accuracy is misleading for imbalanced datasets. With 84% loans being fully paid, a model predicting everything as "paid" gets 84% accuracy while being useless. This project uses ROC-AUC as the primary metric and optimizes the classification threshold based on business cost.

## Business Cost Framework
- Missing a defaulter (False Negative) = ~$11,487 loss (full loan principal)
- Flagging a good borrower (False Positive) = ~$1,378 loss (interest income)
- Optimal threshold: 0.14 (not 0.50) — saves $1.5M in business cost on test data

## Key Results
| Model | ROC-AUC | Default Recall |
|-------|---------|----------------|
| Decision Tree | 0.53 | 0.22 |
| Random Forest | 0.65 | 0.02 |
| RF + SMOTE | 0.65 | 0.23 |
| RF + SMOTE + Optimal Threshold | 0.65 | 0.94 |

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, Imbalanced-learn
- Streamlit (deployment)

## Dataset
LendingClub loan data (2007-2010), 9,578 records
