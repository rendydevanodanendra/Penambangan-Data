import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

# Judul aplikasi
st.title("Prediksi Diabetes - Decision Tree (Model Joblib)")

st.write("Masukkan data pasien untuk prediksi:")

# Form input
pregnancies = st.number_input("Pregnancies", 0.0)
glucose = st.number_input("Glucose", 0.0)
blood_pressure = st.number_input("Blood Pressure", 0.0)
skin_thickness = st.number_input("Skin Thickness", 0.0)
insulin = st.number_input("Insulin", 0.0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0.0)

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    data_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediksi = model.predict(data_input)[0]

    if prediksi == 1:
        st.error("Hasil Prediksi: Positif Diabetes")
    else:
        st.success("Hasil Prediksi: Negatif Diabetes")
