# 🧠 SmartLoan Dashboard

A machine learning-powered Streamlit app that predicts loan default risk using real LendingClub data. It also provides model explainability using SHAP values, allowing users to understand **why** a loan was predicted as risky.

## 🚀 Live App

👉 [Try the App on Streamlit](https://smartloan-dashboard-5hkswlctzwkkega39cwqx9.streamlit.app/)

## 📦 Features

- 📁 Upload CSV with borrower loan application data
- ⚙️ Predict loan default using a trained Random Forest model
- 🔍 Explain model decisions with SHAP (global + local)
- 📊 Visualize prediction distribution via interactive chart

## 🧠 ML Pipeline

- Dataset: LendingClub (filtered and cleaned)
- Model: `RandomForestClassifier`
- Scaling: `StandardScaler`
- Explainabil
