import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os

# display pwd
st.write(os.getcwd())

st.title("Customer Churn Prediction")
st.write("by: [David Saah](https://github.com/davesaah)")

st.subheader("Accuracy metrics", divider="rainbow")
col1, col2, col3, col4 = st.columns(spec=[.5, .5, .5, .15])
col2.metric("Accuracy", "0.81")
col3.metric("AUC Score", "0.86", "0.5")
st.subheader("", divider="rainbow")

# load models
model = load_model("/app/72522025_churning_customers/app/churn_model.h5")
scaler = joblib.load("/app/72522025_churning_customers/app/scaler.pkl")
encoder = joblib.load("/app/72522025_churning_customers/app/encoder.pkl")

def predict():
    # scale numeric values and encode the categorical ones
    num_features = np.array([tenure, monthly_charges, total_charges]).reshape(1, -1)
    num_features_scaled = scaler.transform(num_features)

    cat_features = np.array([contract, internet_service, payment_method]).reshape(1, -1)
    cat_features_encoded = encoder.transform(cat_features)

    # merge the numerical and categorical features
    features = np.concatenate((num_features_scaled, cat_features_encoded.toarray()), axis=1)

    # make prediction
    prediction = model.predict(features)

    if prediction[0] >= 0.5:
        st.error("This customer is likely to churn")
    else:
        st.success("This customer is likely to stay")

# get user input
tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
)

monthly_charges = st.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
)

total_charges = st.number_input(
    "Total Charges ($)",
    min_value=0.0,
)

contract = st.selectbox(
    "Contract",
    options=[
        "Month-to-month",
        "One year",
        "Two year"
    ],
)

payment_method = st.selectbox(
    "Payment Method",
    options=[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ],
)

internet_service = st.selectbox(
    "Internet Service",
    options=[
        "DSL",
        "Fiber optic",
        "No"
    ],
)

col1, col2 = st.columns(spec=[.9, .1])

with col1:
    if st.button("Predict customer churn"):
        predict()

with col2:
    if st.button("Clear"):
        st.session_state = {}