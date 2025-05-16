import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("🛍️ Product Sales Forecasting App")

st.write("Enter the input features below:")

# Input fields for all 19 features
Store_id = st.number_input("Store ID", min_value=1)
Holiday = st.selectbox("Holiday", [0, 1])
Discount = st.selectbox("Discount", ['Yes', 'No'])
Discount_Offered = st.number_input("Discount Offered (%)", min_value=0.0)

Store_Type_S2 = st.selectbox("Store Type S2", [0, 1])
Store_Type_S3 = st.selectbox("Store Type S3", [0, 1])
Store_Type_S4 = st.selectbox("Store Type S4", [0, 1])

Location_Type_L2 = st.selectbox("Location Type L2", [0, 1])
Location_Type_L3 = st.selectbox("Location Type L3", [0, 1])
Location_Type_L4 = st.selectbox("Location Type L4", [0, 1])
Location_Type_L5 = st.selectbox("Location Type L5", [0, 1])

Region_Code_R2 = st.selectbox("Region Code R2", [0, 1])
Region_Code_R3 = st.selectbox("Region Code R3", [0, 1])
Region_Code_R4 = st.selectbox("Region Code R4", [0, 1])

Year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
Month = st.slider("Month", 1, 12)
Day = st.slider("Day", 1, 31)
DayOfWeek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6)
Is_Weekend = st.selectbox("Is Weekend", [0, 1])

# Map 'Yes'/'No' to 1/0
Discount = 1 if Discount == 'Yes' else 0

# Prepare input in correct order
input_features = np.array([[
    Store_id, Holiday, Discount, Year, Month, Day, DayOfWeek, Is_Weekend,
    Discount_Offered, Store_Type_S2, Store_Type_S3, Store_Type_S4,
    Location_Type_L2, Location_Type_L3, Location_Type_L4, Location_Type_L5,
    Region_Code_R2, Region_Code_R3, Region_Code_R4
]])

if st.button("Predict Sales"):
    # Predict
    prediction = model.predict(input_features)
    
    # Inverse transform
    unscaled_prediction = scaler.inverse_transform([[prediction[0]]])[0][0]
    
    st.success(f"📈 Predicted Sales (scaled): {prediction[0]:.2f}")
    st.info(f"💰 Predicted Sales (real): ₹{unscaled_prediction:,.2f}")