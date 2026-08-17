# Loan Default Prediction API

A Machine Learning based FastAPI application that predicts the probability of loan default.

## Current Features

- Loan default prediction using XGBoost
- Data preprocessing pipeline
- Selected important features
- Model saved using Joblib
- FastAPI REST API
- Swagger API documentation
- OAuth2 password authentication
- JWT-based login/authentication
- Protected user profile
- Prediction probability returned by API
- Custom prediction threshold: **0.46**
- Model evaluation with Accuracy, Precision, Recall, F1, ROC-AUC and Average Precision
- MLflow experiment tracking
- ROC and Precision-Recall curves
- MongoDB database connected
- Users collection created
- Predictions collection created

## Model Performance

- Accuracy: ~86.6%
- Precision: ~24.4%
- Recall: ~43.0%
- F1 Score: ~0.31
- ROC-AUC: ~0.76
- Best Threshold: **0.46**

## Tech Stack

Python • FastAPI • XGBoost • Scikit-learn • MongoDB • JWT • OAuth2 • MLflow • Joblib

## Upcoming

- Save predictions in MongoDB
- Prediction history API
- User-specific prediction history
- Delete prediction/history
- Complete API authentication & authorization
- Testing
- Dockerization
- Final deployment
