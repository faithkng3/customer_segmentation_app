#import the necessary libraries
import streamlit as st
import joblib
import numpy as np

# load the pretrained model and scaler
model = joblib.load("new_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# App Title
st.title(" 🎯👥Customer Segmentation Predictor")
st.write("Enter customer details to predict their segment")

# User Inputs
age = st.number_input("Age", 1, 100)
spending_score = st.number_input("Spending Score (1–100)", 1, 100)
work_experience = st.number_input("Work Experience (years)", 0, 50)
family_size = st.number_input("Family Size", 1, 10)
annual_income = st.number_input("Annual Income")

#predict the species of the iris flower
if st.button("Predict customer segment"):
    prediction = model.predict([[age, spending_score, work_experience, family_size,annual_income]])

#reverse encode using mapping(cluster meaning)
    cluster_labels = {
        0: "Low Income, High Spending",
        1: "High Income, Low Spending",
        2: "Average Income, Average Spending",
        3: "High Income, High Spending",
        4: "Low Income, Low Spending"
    }
    st.success(f"Predicted Customer segment is: {cluster_labels[prediction[0]]}")
    