# ============================================================
# 📊 Stock Price Prediction using Support Vector Regression (SVR)
# ============================================================

# -------------------------------
# 📌 Importing Required Libraries
# -------------------------------
import pandas as pd
import numpy as np

# -------------------------------
# 📌 Upload Dataset (Google Colab)
# -------------------------------
from google.colab import files
uploaded = files.upload()   # Upload your CSV file (data.csv)

# -------------------------------
# 📌 Load Dataset
# -------------------------------
dataset = pd.read_csv('data.csv')

# -------------------------------
# 📌 Explore Dataset
# -------------------------------
print("Dataset Shape:", dataset.shape)   # Rows & Columns
print("\nFirst 5 Rows:\n", dataset.head())  # Preview data

# -------------------------------
# 📌 Split Dataset into Features (X) and Target (Y)
# -------------------------------
# X → All columns except last column (Independent Variables)
X = dataset.iloc[:, :-1].values

# Y → Last column (Dependent Variable / Target)
Y = dataset.iloc[:, -1].values

# -------------------------------
# 📌 Split Data into Training & Testing Sets
# -------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.20,      # 20% data for testing
    random_state=0       # For reproducibility
)

# -------------------------------
# 📌 Train Model using SVR
# -------------------------------
from sklearn.svm import SVR

model = SVR()            # Create SVR model
model.fit(x_train, y_train)   # Train model

# -------------------------------
# 📌 Predict on Test Data
# -------------------------------
y_pred = model.predict(x_test)

# -------------------------------
# 📌 Evaluate Model Performance
# -------------------------------
from sklearn.metrics import r2_score, mean_squared_error

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)

# Calculate R² Score
r2score = r2_score(y_test, y_pred)

# -------------------------------
# 📌 Print Results
# -------------------------------
print("\n📊 Model Evaluation Results:")
print("Root Mean Square Error (RMSE):", rmse)
print("R2 Score:", r2score * 100, "%")