import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Dashboard Prediksi Hemoglobin",
    page_icon="🩸",
    layout="wide"
)

# ======================================================
# MEMBACA DATASET
# ======================================================
df = pd.read_csv("anemia.csv")

# Fitur dan Target
X = df[['MCH', 'MCHC', 'MCV']]
y = df['Hemoglobin']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# MEMBUAT MODEL RANDOM FOREST
# ======================================================
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Evaluasi Model
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ======================================================
# DASHBOARD
# ======================================================

st.title("🩸 Dashboard Prediksi Hemoglobin")

st.write("""
Dashboard ini digunakan untuk memprediksi kadar **Hemoglobin (Hb)**
berdasarkan tiga parameter pemeriksaan darah yaitu **MCH**, **MCHC**, dan **MCV**
menggunakan algoritma **Random Forest Regression**.
""")

st.divider()

col1, col2 = st.columns(2)

# ======================================================
# INPUT
# ======================================================

with col1:

    st.subheader("📥 Input Data")

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
        "MCH": [mch],
        "MCHC": [mchc],
        "MCV": [mcv]
    })

    if st.button("🔍 Prediksi Hemoglobin"):

        hasil = rf_model.predict(input_data)[0]

        st.success("## 🩸 Hasil Prediksi")

        st.metric(
            label="Kadar Hemoglobin",
            value=f"{hasil:.2f} g/dL"
        )

        st.subheader("📋 Interpretasi Hasil")

        if hasil < 12:

            st.error("""
### Hemoglobin Rendah

Nilai hemoglobin berada di bawah kisaran normal.

Kemungkinan penyebab:
- Anemia defisiensi besi
- Kekurangan vitamin B12
- Kekurangan asam folat
- Kehilangan darah

Gejala yang mungkin muncul:
- Mudah lelah
- Pusing
- Kulit pucat
- Sesak napas

Disarankan melakukan pemeriksaan laboratorium lebih lanjut.
""")

        elif hasil <= 16:

            st.success("""
### Hemoglobin Normal

Nilai hemoglobin berada pada kisaran normal.

Hal ini menunjukkan kemampuan darah dalam membawa oksigen
kemungkinan berada dalam kondisi yang baik.

Tetap jaga pola hidup sehat dengan:
- Mengonsumsi makanan bergizi
- Istirahat cukup
- Olahraga teratur
- Mengonsumsi makanan kaya zat besi
""")

        else:

            st.warning("""
### Hemoglobin Tinggi

Nilai hemoglobin berada di atas kisaran normal.

Hal ini dapat disebabkan oleh:
- Dehidrasi
- Tinggal di dataran tinggi
- Kebiasaan merokok
- Kondisi medis tertentu

Jika disertai keluhan, lakukan pemeriksaan ke tenaga medis.
""")

        st.info(
            "Hasil prediksi ini merupakan estimasi dari model Machine Learning "
            "dan hanya digunakan untuk tujuan edukasi. "
            "Hasil ini tidak menggantikan diagnosis dokter maupun pemeriksaan laboratorium."
        )

# ======================================================
# EVALUASI MODEL
# ======================================================

with col2:

    st.subheader("📊 Evaluasi Model")

    st.metric("MAE", f"{mae:.3f}")
    st.metric("MSE", f"{mse:.3f}")
    st.metric("RMSE", f"{rmse:.3f}")
    st.metric("R² Score", f"{r2:.3f}")

    st.info("""
Semakin kecil nilai **MAE**, **MSE**, dan **RMSE** maka prediksi model
semakin baik.

Semakin mendekati **1**, nilai **R² Score** menunjukkan model semakin mampu
menjelaskan hubungan antara variabel input dan target.
""")

# ======================================================
# PENJELASAN PARAMETER
# ======================================================

st.divider()

st.header("📖 Penjelasan Parameter")

with st.expander("🩸 MCH (Mean Corpuscular Hemoglobin)"):

    st.markdown("""
**Pengertian**

MCH adalah rata-rata jumlah hemoglobin yang terdapat pada setiap sel darah merah.

**Satuan**
- Pikogram (pg)

**Interpretasi**
- Rendah → Mengindikasikan anemia defisiensi besi.
- Normal → Jumlah hemoglobin dalam eritrosit normal.
- Tinggi → Dapat terjadi pada beberapa jenis anemia.
""")

with st.expander("🩸 MCHC (Mean Corpuscular Hemoglobin Concentration)"):

    st.markdown("""
**Pengertian**

MCHC menunjukkan konsentrasi hemoglobin di dalam sel darah merah.

**Satuan**
- g/dL

**Interpretasi**
- Rendah → Sel darah merah kekurangan hemoglobin.
- Normal → Konsentrasi hemoglobin normal.
- Tinggi → Dapat terjadi pada kondisi medis tertentu.
""")

with st.expander("🩸 MCV (Mean Corpuscular Volume)"):

    st.markdown("""
**Pengertian**

MCV menunjukkan ukuran rata-rata volume sel darah merah.

**Satuan**
- Femtoliter (fL)

**Interpretasi**
- < 80 fL → Anemia mikrositik.
- 80–100 fL → Normal.
- > 100 fL → Anemia makrositik.
""")

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption(
    "Dashboard Prediksi Hemoglobin menggunakan algoritma Random Forest Regression | "
    "Disusun untuk keperluan pembelajaran dan tugas akademik."
)
