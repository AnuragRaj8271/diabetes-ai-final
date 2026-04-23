import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="AI Diabetes Predictor+", layout="wide")
st.title("🧠 AI Diabetes Prediction System (ML + Deep Learning)")
st.markdown("### Logistic Regression + Random Forest + ANN + Explainable AI")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

data = load_data()

# -------------------------------
# Data Cleaning (Important)
# -------------------------------
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    data[col] = data[col].replace(0, data[col].median())

# -------------------------------
# Features
# -------------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# ML Models
# -------------------------------
lr = LogisticRegression()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_acc = accuracy_score(y_test, lr.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# -------------------------------
# Deep Learning Model (ANN)
# -------------------------------
ann = Sequential()
ann.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
ann.add(Dense(8, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=30, batch_size=10, verbose=0)

ann_pred = (ann.predict(X_test) > 0.5).astype(int)
ann_acc = accuracy_score(y_test, ann_pred)

# -------------------------------
# Sidebar Input
# -------------------------------
st.sidebar.header("📥 Enter Patient Data")

preg = st.sidebar.slider("Pregnancies", 0, 15, 1)
glucose = st.sidebar.slider("Glucose", 50, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 40, 150, 70)
skin = st.sidebar.slider("Skin Thickness", 10, 100, 20)
insulin = st.sidebar.slider("Insulin", 15, 300, 80)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5)
age = st.sidebar.slider("Age", 18, 80, 30)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("🔍 Predict"):

    st.subheader("📊 Prediction Result")

    # Logistic Regression
    lr_prob = lr.predict_proba(input_scaled)[0][1]

    # ANN Prediction
    ann_prob = ann.predict(input_scaled)[0][0]

    # Final Output
    final_prob = (lr_prob + ann_prob) / 2

    if final_prob > 0.5:
        st.error("⚠️ High Diabetes Risk")
    else:
        st.success("✅ Low Diabetes Risk")

    st.write(f"### 🔢 Final Risk Score: {round(final_prob*100,2)}%")

    # Risk Level
    if final_prob < 0.3:
        level = "Low"
    elif final_prob < 0.7:
        level = "Medium"
    else:
        level = "High"

    st.write(f"### 📌 Risk Level: {level}")

    # -------------------------------
    # Recommendations
    # -------------------------------
    st.subheader("💡 AI Recommendations")

    rec = []
    if glucose > 140:
        rec.append("Reduce sugar intake")
    if bmi > 30:
        rec.append("Increase physical activity")
    if age > 45:
        rec.append("Regular health checkups")
    if bp > 90:
        rec.append("Monitor blood pressure")

    if rec:
        for r in rec:
            st.write(f"- {r}")
    else:
        st.write("✔️ Maintain healthy lifestyle")

    # -------------------------------
    # Explainable AI (SHAP)
    # -------------------------------
    st.subheader("🔍 Explainable AI")

    explainer = shap.LinearExplainer(lr, X_train)
    shap_values = explainer(input_scaled)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# -------------------------------
# Model Comparison
# -------------------------------
st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Logistic Regression", f"{lr_acc:.2f}")
col2.metric("Random Forest", f"{rf_acc:.2f}")
col3.metric("ANN (Deep Learning)", f"{ann_acc:.2f}")

# -------------------------------
# Heatmap
# -------------------------------
st.subheader("🔥 Feature Correlation")

fig2, ax2 = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)