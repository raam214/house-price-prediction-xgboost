# House Price Prediction using XGBoost

## Overview
This project focuses on predicting house prices using machine learning techniques.
It follows a modular ML pipeline including data preprocessing, feature engineering,
model training, evaluation, and a Streamlit-based web application for predictions.

The goal of this project is to demonstrate a clean, production-style ML workflow
rather than a single notebook-based experiment.

---

## Dataset
- **Dataset**: Ames Housing Dataset
- **Source**: Kaggle – House Prices: Advanced Regression Techniques
- **Note**: Raw dataset files are not included in this repository to keep it clean.
  The dataset is publicly available and can be downloaded from Kaggle.

---

## Project Structure
house-price-prediction-xgboost/
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── src/
│   ├── data_preprocessing.py   # Data loading & cleaning
│   ├── feature_engineering.py  # Feature scaling
│   ├── train_model.py          # XGBoost training pipeline
│   └── evaluate_model.py       # Model evaluation metrics
│
├── notebooks/
│   └── (EDA notebook will be added)
│
├── model/
│   └── xgboost_model.pkl       # Trained model (generated locally)
│
├── requirements.txt
└── README.md
