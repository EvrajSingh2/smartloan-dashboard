# SmartLoan: ML Project with SHAP Explainability

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load and preprocess cleaned LendingClub data
def load_and_preprocess():
    df = pd.read_csv('lending_club_filtered.csv')
    X = df.drop(['loan_status', 'default'], axis=1)
    y = df['default']
    return X, y

# Train and save the model
def train_model():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print("Model Performance:\n", classification_report(y_test, y_pred))
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

# Streamlit Dashboard with SHAP

def main():
    st.title("SmartLoan: Loan Default Predictor with Explainability")
    st.markdown("Upload loan applicant data and get predictions with SHAP-based explanations")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        model_columns = pickle.load(open('model_columns.pkl', 'rb'))

        df_input = df.copy()
        df_input = pd.get_dummies(df_input)
        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[model_columns]

        df_input = df_input.astype(float)
        X_scaled = scaler.transform(df_input)
        df_scaled = pd.DataFrame(X_scaled, columns=model_columns)
        predictions = model.predict(X_scaled)
        df['Default Prediction'] = predictions

        st.subheader("Predictions")
        st.dataframe(df)

        st.subheader("Prediction Breakdown with SHAP")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_scaled, check_additivity=False)
            shap_values_class1 = shap_values[1]  # SHAP values for class 1 (default)

            # Global SHAP Summary
            st.write("### Global Feature Importance")
            fig_summary, ax_summary = plt.subplots()
            shap.summary_plot(
                shap_values_class1,
                features=df_scaled.values,
                feature_names=df_scaled.columns,
                show=False
            )
            st.pyplot(fig_summary)

        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")


        st.subheader("Prediction Distribution")
        st.bar_chart(df['Default Prediction'].value_counts())

if __name__ == '__main__':
    train_model()
    main()




