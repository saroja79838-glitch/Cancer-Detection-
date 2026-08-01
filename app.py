import streamlit as st
import joblib
import pandas as pd
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)
# Load the trained model
model = joblib.load("breast_cancer_model.pkl")
st.title("🩺Breast Cancer Detection System")
st.write("Enter Patient Details")
st.divider()
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean",step=0.01)
    texture_mean = st.number_input("Texture Mean",step=0.01)
    perimeter_mean = st.number_input("Perimeter Mean",step=0.01)
    area_mean = st.number_input("Area Mean",step=0.01)

with col2:
    smoothness_mean = st.number_input("Smoothness Mean",step=0.01)
    compactness_mean = st.number_input("Compactness Mean",step=0.01)
    concavity_mean = st.number_input("Concavity Mean",step=0.01)
    st.divider()

if st.button("🔍Predict Cancer",
use_container_width=True):

    input_data = pd.DataFrame([[

        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean

    ]], columns=[
        'radius_mean',
        'texture_mean',
        'perimeter_mean',
        'area_mean',
        'smoothness_mean',
        'compactness_mean',
        'concavity_mean'
    ])

    prediction = model.predict(input_data)
    probability=model.predict_proba(input_data)

    if prediction[0] == 0:
        
        st.success("Prediction: Benign (No Cancer)")
    else:
        st.error("Prediction: Malignant (Cancer Detected)")
        st.subheader("Prediction Confidence")

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 0:
        st.success("Prediction: Benign (No Cancer)")
    else:
        st.error("Prediction: Malignant (Cancer Detected)")

    st.subheader("Prediction Confidence")
    st.write(f"🟢Benign Probability : {probability[0][0]*100:.2f}%")
    st.write(f"🔴Malignant Probability : {probability[0][1]*100:.2f}%")
    st.progress(float(max(probability[0])))
    st.divider()

st.caption("Developed by Saroja | Breast Cancer Detection Project")