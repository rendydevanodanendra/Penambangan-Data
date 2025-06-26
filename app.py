import streamlit as st
import numpy as np
import joblib

# Load model terbaik hasil training (pastikan file .pkl ada di folder yang sama)
model = joblib.load('random_forest_model.pkl')

st.title('Prediksi Diagnosis Kanker Payudara')
st.write('Masukkan hasil pemeriksaan sel sesuai dataset Breast Cancer Wisconsin:')

# Form input sesuai urutan fitur dataset
clump_thickness = st.slider('Clump Thickness', 1, 10, 1)
uniformity_cell_size = st.slider('Uniformity of Cell Size', 1, 10, 1)
uniformity_cell_shape = st.slider('Uniformity of Cell Shape', 1, 10, 1)
marginal_adhesion = st.slider('Marginal Adhesion', 1, 10, 1)
single_epithelial_cell_size = st.slider('Single Epithelial Cell Size', 1, 10, 1)
bare_nuclei = st.slider('Bare Nuclei', 1.0, 10.0, 1.0)
bland_chromatin = st.slider('Bland Chromatin', 1, 10, 1)
normal_nucleoli = st.slider('Normal Nucleoli', 1, 10, 1)
mitoses = st.slider('Mitoses', 1, 10, 1)

# Tombol Prediksi
if st.button('Prediksi Diagnosis'):
    # Susun data input
    data = np.array([[clump_thickness, uniformity_cell_size, uniformity_cell_shape,
                      marginal_adhesion, single_epithelial_cell_size, bare_nuclei,
                      bland_chromatin, normal_nucleoli, mitoses]])
    
    # Prediksi
    hasil = model.predict(data)

    # Tampilkan hasil
    if hasil[0] == 2:
        st.success('Hasil Prediksi: Tumor Jinak (Benign)')
    elif hasil[0] == 4:
        st.error('Hasil Prediksi: Tumor Ganas (Malignant)')
    else:
        st.warning('Hasil prediksi tidak diketahui.')
