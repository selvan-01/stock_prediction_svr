# ============================================================
# 📊 Stock Prediction using Support Vector Regression (SVR)
# 🚀 Advanced Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

# -------------------------------
# 🎨 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# 🏷️ Title
# -------------------------------
st.title("📈 Stock Price Prediction using SVR")
st.markdown("Upload your dataset and predict stock prices using Machine Learning")

# -------------------------------
# 📁 File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    
    # -------------------------------
    # 📊 Load Dataset
    # -------------------------------
    dataset = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(dataset.head())

    st.write("Shape of dataset:", dataset.shape)

    # -------------------------------
    # 🎯 Select Target Column
    # -------------------------------
    target_column = st.selectbox(
        "Select Target Column (What you want to predict)",
        dataset.columns
    )

    # -------------------------------
    # 🧩 Feature & Target Split
    # -------------------------------
    X = dataset.drop(columns=[target_column]).values
    Y = dataset[target_column].values

    # -------------------------------
    # ⚙️ Train-Test Split
    # -------------------------------
    test_size = st.slider("Select Test Size (%)", 10, 40, 20)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=test_size / 100,
        random_state=42
    )

    # -------------------------------
    # 🤖 Train Model Button
    # -------------------------------
    if st.button("🚀 Train Model"):
        
        model = SVR()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        # -------------------------------
        # 📊 Evaluation Metrics
        # -------------------------------
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Model Trained Successfully!")

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MSE", f"{mse:.2f}")
        col3.metric("R2 Score", f"{r2*100:.2f}%")

        # -------------------------------
        # 📈 Visualization
        # -------------------------------
        st.subheader("📈 Actual vs Predicted")

        fig = plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted Stock Prices")
        st.pyplot(fig)

        # -------------------------------
        # 🔮 Prediction Section
        # -------------------------------
        st.subheader("🔮 Make Custom Prediction")

        input_data = []

        for col in dataset.drop(columns=[target_column]).columns:
            value = st.number_input(f"Enter value for {col}")
            input_data.append(value)

        if st.button("Predict Price"):
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)

            st.success(f"💰 Predicted Stock Price: {prediction[0]:.2f}")

else:
    st.info("👆 Upload a CSV file to get started")