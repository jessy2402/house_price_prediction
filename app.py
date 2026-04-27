import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Singapore Flat Price Prediction", layout="wide")

st.title("🏠 Singapore Resale Flat Price Prediction")

# -------------------------------
# LOAD MODEL FILES
# -------------------------------
model = pickle.load(open("DTR_model.pkl", "rb"))
X_test = pickle.load(open("X_test.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("Enter Flat Details")

month = st.sidebar.slider("Month", 1, 12, 5)
storey_range = st.sidebar.slider("Storey Range (Encoded)", 0, 10, 3)
floor_area_sqm = st.sidebar.slider("Floor Area (sqm)", 30, 200, 80)
lease_commence_date = st.sidebar.slider("Lease Year", 1966, 2023, 2000)
year = st.sidebar.slider("Year", 1990, 2023, 2020)

# -------------------------------
# PREDICTION
# -------------------------------
input_data = np.array([[month, storey_range, floor_area_sqm, lease_commence_date, year]])
prediction = model.predict(input_data)[0]

st.subheader("💰 Predicted Resale Price")
st.success(f"SGD {prediction:,.2f}")

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
st.subheader("📈 Model Performance")

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

col1, col2, col3 = st.columns(3)

col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("R² Score", f"{r2:.2f}")
col3.metric("Accuracy (%)", f"{accuracy:.2f}%")

# -------------------------------
# GRAPH 1: LINE GRAPH
# -------------------------------
st.subheader("📊 Graph 1: Actual vs Predicted Prices")

fig1, ax1 = plt.subplots()
ax1.plot(y_test.values[:50], label="Actual")
ax1.plot(y_pred[:50], label="Predicted")
ax1.set_xlabel("Samples")
ax1.set_ylabel("Price")
ax1.set_title("Actual vs Predicted Trend")
ax1.legend()

st.pyplot(fig1)

# -------------------------------
# GRAPH 2: BAR GRAPH
# -------------------------------
st.subheader("📊 Graph 2: Error Metrics Comparison")

mae = np.mean(np.abs(y_test - y_pred))

fig2, ax2 = plt.subplots()
metrics = ["MAE", "RMSE"]
values = [mae, rmse]

ax2.bar(metrics, values)
ax2.set_ylabel("Error Value")
ax2.set_title("Error Metrics")

st.pyplot(fig2)