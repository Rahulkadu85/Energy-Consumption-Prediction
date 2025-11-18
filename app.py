#streamlit library
import streamlit as st
import pickle
import pandas as pd

# Load model, scaler, encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.title("Energy Consumption & Bill Predictor")

# User Inputs
building_type = st.selectbox("Building Type", ["Residential", "Commercial", "Industrial"])
square_footage = st.number_input("Square Footage", min_value=0)
num_occupants = st.number_input("Number of Occupants", min_value=0)
appliances_used = st.number_input("Appliances Used", min_value=0)
avg_temp = st.number_input("Average Temperature (°C)")
day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])

if st.button("Predict"):
    new_data = pd.DataFrame({
        "Building Type": [building_type],
        "Square Footage": [square_footage],
        "Number of Occupants": [num_occupants],
        "Appliances Used": [appliances_used],
        "Average Temperature": [avg_temp],
        "Day of Week": [day_of_week]
    })

    # Encode categorical columns
    cat_col = new_data.select_dtypes(include=['object'])
    for col in cat_col:
        new_data[col] = encoder.fit_transform(new_data[col])

    # Scale and predict
    scaled_data = scaler.transform(new_data)
    consumption = model.predict(scaled_data)
    cost_per_unit = 10
    bill = consumption * cost_per_unit

    st.success(f"Predicted Energy Consumption: {consumption[0]:.2f} units")
    st.success(f"Estimated Bill: ₹{bill[0]:.2f}")