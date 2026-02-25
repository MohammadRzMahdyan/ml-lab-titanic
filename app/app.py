import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os, sys
import imghdr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers.custom_transformers import *

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../outputs/models/titanic_xgb_model.pkl")

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Titanic Survival App",
    page_icon="🚢",
    layout="wide"
)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ------------------ Sidebar Inputs ------------------
with st.sidebar:
    st.header("Passenger Info")
    pclass = st.selectbox("Class", [1,2,3])
    gender = st.selectbox("Gender", ["Male","Female"])
    age = st.slider("Age", 0, 100, 25)
    fare = st.number_input("Fare", 0.0, 500.0, 50.0)
    sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
    parch = st.number_input("Parents/Children", 0, 10, 0)
    predict_button = st.button("Predict Survival")

# ------------------ Prepare Input for Pipeline ------------------
def make_input():
    return pd.DataFrame({
        'pclass': [pclass],
        'name': ["unknown"], 
        'sex': [gender],
        'sibsp': [sibsp],
        'parch': [parch],
        'ticket': ["unknown"],
        'fare': [fare],
        'age': [age],
        'embarked': [np.nan] 
    })

# ------------------ Main Layout ------------------
st.header("🚢 Titanic Survival Predictor")

if predict_button:
    X_input = make_input()
    proba = float(model.predict_proba(X_input)[:,1][0])

    st.subheader("📈 Prediction Result")
    st.progress(proba)
    if proba > 0.4:
        st.success(f"High Chance of Survival 🎉 ({proba:.2f})")
    else:
        st.error(f"Low Chance of Survival 💀 ({proba:.2f})")
else:
    st.info("Fill the passenger info in the sidebar and click Predict.")