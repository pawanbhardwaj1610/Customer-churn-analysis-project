import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Dashboard", layout="centered")

st.title("📊 Customer Churn Prediction Dashboard")

# ===============================
# INPUTS
# ===============================
st.header("🧾 Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.number_input("Monthly Charges", value=70.0)
TotalCharges = st.number_input("Total Charges", value=800.0)

# ===============================
# PREDICT
# ===============================
if st.button("🚀 Predict"):

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    if response.status_code == 200:
        result = response.json()

        prediction = result["prediction"]
        prob = result["probability"] * 100

        # ===============================
        # COLOR RESULT
        # ===============================
        if prediction == "Churn":
            st.error(f"⚠️ High Risk of Churn ({prob:.2f}%)")
        else:
            st.success(f"✅ Low Risk ({prob:.2f}%)")

        # ===============================
        # GAUGE CHART
        # ===============================
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if prob > 50 else "green"},
            }
        ))

        st.plotly_chart(fig)

        # ===============================
        # BAR CHART
        # ===============================
        fig2 = go.Figure(data=[
            go.Bar(name='No Churn', x=['Result'], y=[100 - prob]),
            go.Bar(name='Churn', x=['Result'], y=[prob])
        ])

        fig2.update_layout(barmode='stack')
        st.plotly_chart(fig2)

        # ===============================
        # JSON DETAILS
        # ===============================
        st.subheader("🔍 Details")
        st.json(result)

    else:
        st.error("API Error")