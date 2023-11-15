import streamlit as st
import joblib
from tensorflow.keras.models import load_model

st.title("Customer Churn Prediction")
st.write("by: [David Saah](https://github.com/davesaah)")

st.subheader("Accuracy metrics", divider="rainbow")
col1, col2, col3, col4 = st.columns(spec=[.5, .5, .5, .15])
col2.metric("Accuracy", "0.81")
col3.metric("AUC Score", "0.86", "0.5")
st.subheader("", divider="rainbow")

# load models
model = load_model("../models/churn_model.h5")
scaler = joblib.load("../models/scaler.pkl")
encoder = joblib.load("../models/encoder.pkl")

def predict():
    pass

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
        "Two years"
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