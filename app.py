
import streamlit as st
import pandas as pd
import numpy as np # Added numpy import
from google.colab import drive
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Mount Google Drive (if not already mounted) ---
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    st.write("Google Drive is already mounted or failed to mount.")


# --- Load Data and Model ---
df = pd.read_csv('/content/drive/MyDrive/Big Data/DATASET/anemia.csv')

X = df[['MCH', 'MCHC', 'MCV']]
y = df['Hemoglobin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)


# --- Dashboard Layout ---
st.set_page_config(layout="wide")
st.title("Dashboard Prediksi Hemoglobin Interaktif")

st.write("Selamat datang di dasbor prediksi kadar Hemoglobin. Gunakan slider di bawah untuk menyesuaikan nilai fitur dan melihat prediksi secara real-time.")

st.markdown("--- --- --- ---")

col1, col2 = st.columns(2)

with col1:
    st.header("Masukkan Nilai Fitur")

    mch_mean_val = df['MCH'].mean()
    mchc_mean_val = df['MCHC'].mean()
    mcv_mean_val = df['MCV'].mean()

    mch_input = st.slider('MCH', min_value=14.8, max_value=30.8, value=mch_mean_val, step=0.1)
    mchc_input = st.slider('MCHC', min_value=26.5, max_value=34.1, value=mchc_mean_val, step=0.1)
    mcv_input = st.slider('MCV', min_value=68.3, max_value=102.8, value=mcv_mean_val, step=0.1)

    input_data = pd.DataFrame([[mch_input, mchc_input, mcv_input]], columns=['MCH', 'MCHC', 'MCV'])

    if st.button('Prediksi Hemoglobin'):
        predicted_hemoglobin = rf_model.predict(input_data)[0]
        st.success(f"### Prediksi Kadar Hemoglobin: **{predicted_hemoglobin:.3f}**")
    else:
        predicted_hemoglobin = rf_model.predict(input_data)[0]
        st.info(f"### Prediksi Kadar Hemoglobin: **{predicted_hemoglobin:.3f}** (geser slider untuk update)")

with col2:
    st.header("Evaluasi Model Random Forest")
    st.markdown(f"- **MAE:** {mae_rf:.3f}")
    st.markdown(f"- **MSE:** {mse_rf:.3f}")
    st.markdown(f"- **RMSE:** {rmse_rf:.3f}")
    st.markdown(f"- **R2 Score:** {r2_rf:.3f}")
    st.warning("*R2 Score yang rendah menunjukkan model ini belum terlalu akurat dalam memprediksi Hemoglobin.*")

st.markdown("--- --- --- ---")
st.write("\n\nIni adalah contoh dasbor. Anda dapat menambahkan lebih banyak visualisasi, metrik, atau fitur lain sesuai kebutuhan.")
