import streamlit as st
import joblib
import numpy as np

# Load the model using a corrected path
model = joblib.load("model/cust_churn.pkl")  # Use forward slashes for paths
labels = ['No', 'Yes']

st.title("Customer Churn in Telecom - Prediction APP")

# Use selectbox for categorical inputs
OnSec = st.selectbox("Provide Online Security? (Yes, No, No Internet Service):", ['Yes', 'No', 'No Internet Service'])
OnBackup = st.selectbox("Provide Online Backup? (Yes, No, No Internet Service):", ['Yes', 'No', 'No Internet Service'])
DevPro = st.selectbox("Provide Device Protection? (Yes, No, No Internet Service):", ['Yes', 'No', 'No Internet Service'])
TechSup = st.selectbox("Provide Tech Support? (Yes, No, No Internet Service):", ['Yes', 'No', 'No Internet Service'])

# Map string inputs to numerical values
def map_input_to_numeric(input_value):
    if input_value == 'Yes':
        return 1
    elif input_value == 'No':
        return 0
    else:  # Handle 'No Internet Service'
        return 2  # Or another numeric value that your model can handle

# Prepare input data for model prediction
input_data = np.array([
    map_input_to_numeric(OnSec),
    map_input_to_numeric(OnBackup),
    map_input_to_numeric(DevPro),
    map_input_to_numeric(TechSup)
]).reshape(1, -1)

# Handle prediction button
if st.button("Prediction!"):
    prediction = model.predict(input_data)
    st.write("Prediction:", labels[prediction[0]])

proba = model.predict_proba(input_data)
st.write("Probabilities:", proba)
