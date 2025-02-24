import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("day.csv")
df['dteday'] = pd.to_datetime(df['dteday'])

# Drop unnecessary columns
df.drop(columns=["instant", "dteday"], inplace=True)

# Convert categorical variables
df = pd.get_dummies(df, columns=["season", "mnth", "weekday", "weathersit"], drop_first=True)

# Split data for casual and registered separately
X = df.drop(columns=["cnt", "casual", "registered"])
y_casual = df["casual"]
y_registered = df["registered"]
X_train, X_test, y_casual_train, y_casual_test, y_registered_train, y_registered_test = train_test_split(
    X, y_casual, y_registered, test_size=0.2, random_state=42
)

# Train Random Forest models for casual and registered users
rf_casual = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_registered = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_casual.fit(X_train, y_casual_train)
rf_registered.fit(X_train, y_registered_train)

# Predictions
y_casual_pred = rf_casual.predict(X_test)
y_registered_pred = rf_registered.predict(X_test)
y_pred_total = y_casual_pred + y_registered_pred

r2_casual = r2_score(y_casual_test, y_casual_pred)
r2_registered = r2_score(y_registered_test, y_registered_pred)
r2_total = r2_score(y_casual_test + y_registered_test, y_pred_total)

# Anomaly detection
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = isolation_forest.fit_predict(X)
anomaly_data = df[df['anomaly'] == -1]
anomaly_percentage = anomaly_data.shape[0] / df.shape[0] * 100

# Streamlit UI
st.title("🚲 Bike Rental Prediction & Anomaly Detection")
st.markdown("### 📊 Model Performance")
st.metric("R² Score (Total)", f"{r2_total:.4f}")
st.metric("R² Score (Casual)", f"{r2_casual:.4f}")
st.metric("R² Score (Registered)", f"{r2_registered:.4f}")

# Line plot
st.markdown("### 📅 Bike Rentals Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=df.index, y=df["cnt"], ax=ax)
st.pyplot(fig)

# Anomalies
st.markdown("### ⚠️ Anomaly Detection")
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(df.index, df['cnt'], label='Normal Rentals', color='blue')
plt.scatter(anomaly_data.index, anomaly_data['cnt'], label='Anomalous Rentals', color='red', zorder=5)
st.pyplot(fig)
st.write(f"🚨 Percentage of anomalies: {anomaly_percentage:.2f}%")

# User Input Prediction
st.markdown("### 🔮 Predict Rentals")
input_data = {}
input_data["yr"] = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
input_data["mnth"] = st.selectbox("Month", list(range(1, 13)))
input_data["weekday"] = st.selectbox("Weekday", list(range(0, 7)), format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][x])
input_data["holiday"] = st.radio("Holiday", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
input_data["workingday"] = st.radio("Working Day", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
input_data["weathersit"] = st.selectbox("Weather Situation", [1, 2, 3, 4], format_func=lambda x: ["Clear", "Mist/Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"][x-1])
input_data["temp"] = st.slider("Temperature (Normalized)", float(df["temp"].min()), float(df["temp"].max()), float(df["temp"].mean()))
input_data["atemp"] = st.slider("Feeling Temperature (Normalized)", float(df["atemp"].min()), float(df["atemp"].max()), float(df["atemp"].mean()))
input_data["hum"] = st.slider("Humidity", float(df["hum"].min()), float(df["hum"].max()), float(df["hum"].mean()))
input_data["windspeed"] = st.slider("Windspeed", float(df["windspeed"].min()), float(df["windspeed"].max()), float(df["windspeed"].mean()))
input_df = pd.DataFrame([input_data])

# Ensure column order matches training data
input_df = input_df.reindex(columns=X.columns, fill_value=0)
if st.button("🔍 Predict"):
    casual_prediction = rf_casual.predict(input_df)[0]
    registered_prediction = rf_registered.predict(input_df)[0]
    total_prediction = casual_prediction + registered_prediction
    
    st.success(f"🚴 Predicted Rentals: {int(total_prediction)}")
    st.write(f"👥 Registered Users: {int(registered_prediction)}")
    st.write(f"🛴 Casual Users: {int(casual_prediction)}")
