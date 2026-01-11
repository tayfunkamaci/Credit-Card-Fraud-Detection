import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fraud Decision Engine",
    page_icon="💳",
    layout="wide"
)

# =====================================================
# LOAD MODEL PACKAGE
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")

pkg = load_model()

model = pkg["model"]
features = pkg["features"]
threshold = pkg["threshold"]

# PCA ortalamaları (train setten)
pca_means = pkg.get(
    "pca_means",
    {f"V{i}": 0.0 for i in range(1, 29)}
)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 style='text-align:center;'>💳 Fraud Decision Engine</h1>
    <h4 style='text-align:center;color:gray;'>
    Policy-driven • Model + Business Rules
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("🧾 İşlem Bilgileri")

amount = st.sidebar.number_input(
    "İşlem Tutarı (₺)",
    min_value=1.0,
    max_value=200000.0,
    value=500.0,
    step=100.0
)

time_diff = st.sidebar.selectbox(
    "Son işlemden geçen süre (sn)",
    [5, 10, 15, 30, 60, 300, 600]
)

hour = st.sidebar.slider("İşlem Saati", 0, 23, 14)

is_night = 1 if hour >= 22 or hour < 6 else 0

st.sidebar.markdown("---")
st.sidebar.caption("Threshold eğitim aşamasında cost-based optimize edilmiştir.")

# =====================================================
# FEATURE ENGINEERING (MAIN.PY İLE UYUMLU)
# =====================================================
data = {
    "Amount": amount,
    "Amount_Log": np.log1p(amount),
    "Time_Diff": time_diff,
    "Hour": hour,
    "Is_Night": is_night
}

# PCA feature'ları (gerçek hayatta yok → ortalama ile doldur)
for i in range(1, 29):
    data[f"V{i}"] = pca_means.get(f"V{i}", 0.0)

# PCA agregasyonları (EKSİKSİZ)
pca_vals = np.array([data[f"V{i}"] for i in range(1, 29)])

data["PCA_Abs_Mean"] = np.mean(np.abs(pca_vals))
data["PCA_Pos_Sum"] = np.sum(pca_vals[pca_vals > 0])
data["PCA_Neg_Sum"] = np.sum(pca_vals[pca_vals < 0])

# DataFrame + feature order
df = pd.DataFrame([data])[features]

# =====================================================
# POLICY / KARAR MANTIĞI
# =====================================================
def decision_policy(amount, time_diff, is_night, model_proba, threshold):
    """
    main.py'den çıkan analizlere dayalı karar politikası
    """

    # Risk flag'leri
    high_amount = amount >= 50000
    medium_amount = amount >= 20000
    fast_tx = time_diff <= 10
    night_tx = is_night == 1
    high_model_risk = model_proba >= threshold

    # Risk skoru
    risk_score = sum([
        high_amount,
        fast_tx,
        night_tx,
        high_model_risk
    ])

    # Karar
    if risk_score >= 3:
        return "BLOCK", "Çoklu yüksek risk faktörü"
    elif risk_score == 2:
        return "CHALLENGE", "Şüpheli işlem – ek doğrulama gerekli"
    else:
        return "ALLOW", "Normal işlem profili"

# =====================================================
# PREDICTION
# =====================================================
if st.button("🚀 Analiz Et"):

    with st.spinner("İşlem analiz ediliyor..."):
        time.sleep(1)

        proba = model.predict_proba(df)[0][1]

    decision, explanation = decision_policy(
        amount=amount,
        time_diff=time_diff,
        is_night=is_night,
        model_proba=proba,
        threshold=threshold
    )

    # =================================================
    # OUTPUT
    # =================================================
    st.markdown("---")

    c1, c2 = st.columns(2)
    c1.metric("Fraud Olasılığı", f"{proba:.2%}")
    c2.metric("Karar", decision)

    if decision == "ALLOW":
        st.success("🟢 NORMAL İŞLEM – İşlem onaylandı")
    elif decision == "CHALLENGE":
        st.warning("🟡 ŞÜPHELİ – Ek doğrulama gerekli")
    else:
        st.error("🔴 FRAUD – İşlem engellendi")

    # =================================================
    # DECISION EXPLANATION
    # =================================================
    st.markdown("### 🧠 Karar Gerekçesi")
    st.write(explanation)

    if amount >= 50000:
        st.write("• Çok yüksek tutar")
    elif amount >= 20000:
        st.write("• Orta-yüksek tutar")

    if time_diff <= 10:
        st.write("• Çok kısa sürede ardışık işlem")

    if is_night:
        st.write("• Gece saati işlemi")

    if proba >= threshold:
        st.write("• Model yüksek fraud olasılığı verdi")

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <hr>
    <p style='text-align:center;color:gray;'>
    Fraud Detection • Policy-driven Decision Engine
    </p>
    """,
    unsafe_allow_html=True
)
