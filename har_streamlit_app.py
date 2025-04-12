import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import io

st.set_page_config(page_title="Human Activity Recognition App", layout="wide")
st.title("🏃 Human Activity Recognition (HAR) using ML")

uploaded_file = st.file_uploader("📂 Upload HAR CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Label Encode Activity
    le = LabelEncoder()
    df['Activity'] = le.fit_transform(df['Activity'])

    # Correlation Heatmap
    st.subheader("📊 Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap='coolwarm')
    st.pyplot(fig)

    # Features and labels
    X = df.drop('Activity', axis=1)
    y = df['Activity']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X_scaled, y)

    # PCA Visualization
    st.subheader("🔎 PCA - Explained Variance")
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_selected)
    fig2, ax2 = plt.subplots()
    ax2.plot(np.cumsum(pca.explained_variance_ratio_))
    ax2.set_xlabel("Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    st.pyplot(fig2)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    # Train XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    # Show confusion matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig3)

    # Accuracy
    st.success(f"✅ XGBoost Model Accuracy: {xgb.score(X_test, y_test) * 100:.2f}%")

    # Downloadable model prediction CSV
    results_df = pd.DataFrame({"Actual": le.inverse_transform(y_test), "Predicted": le.inverse_transform(y_pred)})
    csv = results_df.to_csv(index=False).encode()
    st.download_button("📥 Download Predictions as CSV", csv, "predictions.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to get started.")