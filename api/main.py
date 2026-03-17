from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn API 🚀")

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ===============================
# FEATURE ORDER (VERY IMPORTANT)
# ===============================
FEATURE_NAMES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

# ===============================
# ENCODING MAPS (MATCH TRAINING)
# ===============================
binary_map = {"Yes": 1, "No": 0}

gender_map = {"Male": 1, "Female": 0}

multi_map = {
    "No": 0,
    "Yes": 1,
    "No phone service": 2
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

service_map = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

# ===============================
# INPUT SCHEMA (USER FRIENDLY)
# ===============================
class InputData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ===============================
# HOME ROUTE
# ===============================
@app.get("/")
def home():
    return {"message": "User-friendly Churn Prediction API 🚀"}

# ===============================
# ENCODING FUNCTION
# ===============================
def encode_input(data):
    return {
        "gender": gender_map[data.gender],
        "SeniorCitizen": data.SeniorCitizen,
        "Partner": binary_map[data.Partner],
        "Dependents": binary_map[data.Dependents],
        "tenure": data.tenure,
        "PhoneService": binary_map[data.PhoneService],
        "MultipleLines": multi_map[data.MultipleLines],
        "InternetService": internet_map[data.InternetService],
        "OnlineSecurity": service_map[data.OnlineSecurity],
        "OnlineBackup": service_map[data.OnlineBackup],
        "DeviceProtection": service_map[data.DeviceProtection],
        "TechSupport": service_map[data.TechSupport],
        "StreamingTV": service_map[data.StreamingTV],
        "StreamingMovies": service_map[data.StreamingMovies],
        "Contract": contract_map[data.Contract],
        "PaperlessBilling": binary_map[data.PaperlessBilling],
        "PaymentMethod": payment_map[data.PaymentMethod],
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges
    }

# ===============================
# PREDICT ROUTE
# ===============================
@app.post("/predict")
def predict(data: InputData):
    try:
        # Encode input
        encoded_data = encode_input(data)

        # Convert to DataFrame
        df = pd.DataFrame([encoded_data])
        df = df[FEATURE_NAMES]

        # Scale
        scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "probability": float(probability)
        }

    except Exception as e:
        return {"error": str(e)}