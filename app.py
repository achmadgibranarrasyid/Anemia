import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi halaman (harus di paling atas)
st.set_page_config(
    page_title="Dashboard Prediksi Hemoglobin",
    page_icon="🩸",
    layout="wide"
)

# Membaca dataset
df = pd.read_csv("anemia.csv")

# Menentukan fitur dan target
X = df[['MCH', 'MCHC', 'MCV']]
y = df['Hemoglobin']

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Melatih model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Evaluasi model
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# =======================
# Dashboard
# =======================

st.title("🩸 Dashboard Prediksi Hemoglobin")

st.write(
    "Masukkan nilai MCH, MCHC, dan MCV untuk memprediksi kadar Hemoglobin."
)

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("Input Data")

    mch = st.slider(
        "MCH",
        float(df["MCH"].min()),
        float(df["MCH"].max()),
        float(df["MCH"].mean())
    )

    mchc = st.slider(
        "MCHC",
        float(df["MCHC"].min()),
        float(df["MCHC"].max()),
        float(df["MCHC"].mean())
    )

    mcv = st.slider(
        "MCV",
        float(df["MCV"].min()),
        float(df["MCV"].max()),
        float(df["MCV"].mean())
    )

    input_data = pd.DataFrame({
        "MCH":[mch],
        "MCHC":[mchc],
        "MCV":[mcv]
    })

    if st.button("Prediksi"):

        hasil = rf_model.predict(input_data)[0]

        st.success(
            f"Prediksi Hemoglobin : **{hasil:.2f}**"
        )

with col2:

    st.subheader("Evaluasi Model")

    st.metric("MAE", f"{mae:.3f}")
    st.metric("MSE", f"{mse:.3f}")
    st.metric("RMSE", f"{rmse:.3f}")
    st.metric("R² Score", f"{r2:.3f}")

st.divider()

st.caption("Dashboard Prediksi Hemoglobin menggunakan Random Forest Regression")
