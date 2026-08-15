# Loan Default Prediction - End-to-End MLOps Project

An end-to-end Machine Learning and MLOps project for predicting the probability of loan default using the Home Credit dataset.

The project covers the complete ML workflow from data ingestion and preprocessing to model training, evaluation, experiment tracking, data versioning, and the initial FastAPI backend setup.

---

## 🚀 Project Overview

Loan default prediction is a classification problem where the goal is to identify customers who are likely to default on a loan.

The dataset is highly imbalanced, so the project does not rely only on accuracy. Multiple evaluation metrics such as Precision, Recall, F1-score, ROC-AUC and Average Precision are used.

The project is designed with an MLOps-oriented architecture using:

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- DVC
- MLflow
- FastAPI
- MongoDB
- JWT Authentication
- Docker (planned)
- GitHub Actions CI/CD (planned)

---

# 📁 Project Structure

```text
Loan-Detection/
│
├── app/
│   ├── config.py
│   ├── database.py
│   ├── main.py
│   ├── router.py
│   ├── schemas.py
│   └── services.py
│
├── data/
│   ├── raw/
│   ├── selected/
│   ├── processed/
│   ├── model/
│   ├── metrics/
│   └── plots/
│
├── experiments/
│
├── logs/
│
├── mlruns/
│
├── src/
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── .dvc/
├── .dvcignore
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── mlflow.db
├── requirements.txt
└── README.md
