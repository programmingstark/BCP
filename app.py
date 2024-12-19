
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and feature names
model = joblib.load("diagnosis_model.pkl")
feature_names = joblib.load("feature_names1.pkl")

# Define the app layout
st.title("Breast Cancer Diagnosis Prediction")
st.write("Provide the following tumor features to predict the diagnosis (Malignant or Benign):")

# Create input fields for all required features
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(feature.replace("x.", "").replace("_", " ").capitalize(), value=0.0)

# Prediction logic
if st.button("Predict Diagnosis"):
    # Convert inputs into a DataFrame
    input_data = pd.DataFrame([inputs.values()], columns=feature_names)
    
    # Preprocess the inputs (standardize)
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)  # Assuming the model expects standardized inputs
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    
    # Display the result
    st.success(f"The predicted diagnosis is: {result}")
