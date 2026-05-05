from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

from utils.helpers import risk_label

app = FastAPI()

model = pickle.load(open("models/model.pkl","rb"))
scaler = pickle.load(open("models/scaler.pkl","rb"))

# ✅ Input validation
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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


@app.get("/")
def home():
    return {"message": "Churn API Running"}


@app.post("/predict")
def predict(data: CustomerInput):

    input_df = pd.DataFrame([data.dict()])

    # Feature engineering
    input_df["TotalCharges"] = input_df["MonthlyCharges"] * input_df["tenure"]
    input_df["AvgMonthlySpend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
    input_df["IsMonthly"] = (input_df["Contract"] == "Month-to-month").astype(int)
    input_df["HighValueCustomer"] = (input_df["MonthlyCharges"] > 80).astype(int)

    # Encode
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = input_df[col].astype("category").cat.codes

    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_probability": round(float(prob), 2),
        "churn_risk": risk_label(prob)
    }