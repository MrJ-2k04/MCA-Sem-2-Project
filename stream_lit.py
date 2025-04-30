import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("Cleaned_Bike_Data.csv")

# Encode categorical variables
label_encoders = {}
for col in ['model', 'city', 'owner', 'brand']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop(['price'], axis=1)
y = df['price']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🚲 Bike Price Prediction App")
st.write("Designed by JAY SONI AND MOHIT JAIN")
st.write("Enter bike details below to predict its market price.")

# User input fields
model_input = st.selectbox("Model", label_encoders['model'].classes_)
city_input = st.selectbox("City", label_encoders['city'].classes_)
kms_driven_input = st.number_input("KMs Driven", min_value=0, step=100)
owner_input = st.selectbox("Owner Type", label_encoders['owner'].classes_)
age_input = st.slider("Age (Years)", 0, 30, 3)
power_input = st.slider("Power (CC)", 50, 1000, 100)
brand_input = st.selectbox("Brand", label_encoders['brand'].classes_)

# Encode inputs
input_data = pd.DataFrame({
    'model': [label_encoders['model'].transform([model_input])[0]],
    'city': [label_encoders['city'].transform([city_input])[0]],
    'kms_driven': [kms_driven_input],
    'owner': [label_encoders['owner'].transform([owner_input])[0]],
    'age': [age_input],
    'power': [power_input],
    'brand': [label_encoders['brand'].transform([brand_input])[0]],
    'owner_encoded': [label_encoders['owner'].transform([owner_input])[0]]
})

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Bike Price: ₹{int(prediction):,}")
