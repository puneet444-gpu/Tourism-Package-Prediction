
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# ------------------------------
# Model Configuration
# ------------------------------
MODEL_REPO = "puneet44/tourism-package-predictor-model"
MODEL_FILE = "model.joblib"

@st.cache_resource
def load_model():
    """Load model from Hugging Face Hub and cache it for reuse."""
    path = hf_hub_download(MODEL_REPO, MODEL_FILE, repo_type="model")
    return joblib.load(path)

model = load_model()

# ------------------------------
# Streamlit Page Title
# ------------------------------
st.title("Wellness Tourism Package Predictor")

# ------------------------------
# Function: User Inputs
# ------------------------------
def user_inputs():
    """Collect input features from the user via Streamlit widgets."""
    return {
        "Age": st.number_input("Age", 18, 100, 30),
        "TypeofContact": st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"]),
        "CityTier": st.selectbox("City Tier", [1, 2, 3]),
        "DurationOfPitch": st.number_input("Duration Of Pitch", 1, 60, 10),
        "Occupation": st.selectbox("Occupation", ["Salaried", "Free Lancer"]),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "NumberOfPersonVisiting": st.number_input("Persons Visiting", 1, 10, 2),
        "ProductPitched": st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard"]),
        "PreferredPropertyStar": st.selectbox("Hotel Star", [3, 4, 5]),
        "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
        "NumberOfTrips": st.number_input("Trips Per Year", 0, 20, 2),
        "Passport": st.selectbox("Passport", [0, 1]),
        "PitchSatisfactionScore": st.slider("Pitch Satisfaction", 1, 5, 3),
        "OwnCar": st.selectbox("Own Car", [0, 1]),
        "NumberOfChildrenVisiting": st.number_input("Children Visiting", 0, 5, 0),
        "Designation": st.selectbox("Designation", ["Executive", "Manager"]),
        "MonthlyIncome": st.number_input("Monthly Income", 5000, 200000, 30000)
    }

# ------------------------------
# Collect Inputs and Create DataFrame
# ------------------------------
inputs = user_inputs()
input_df = pd.DataFrame([inputs])

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {'Will Purchase' if pred else 'Will Not Purchase'}")
    st.write(f"Probability: {prob:.2f}")
