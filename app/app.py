import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction App")
st.write("Enter the details below to predict house price.")

# load trained model
model = joblib.load("model/xgboost_model.pkl")

# input fields (example features)
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)
feature_4 = st.number_input("Feature 4", value=0.0)
feature_5 = st.number_input("Feature 5", value=0.0)

if st.button("Predict Price"):
    features = np.array(
        [[feature_1, feature_2, feature_3, feature_4, feature_5]])
    prediction = model.predict(features)

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
