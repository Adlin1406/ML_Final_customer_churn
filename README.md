# 📊 Customer Churn Prediction (ML Project)

## 📌 Problem
Predict whether a telecom customer will churn so the business can take proactive retention actions.

---

## 📊 Dataset
Customer data includes:
- Demographics (gender, senior citizen)
- Account details (tenure, contract)
- Services (internet, support, streaming)
- Billing (monthly & total charges)

Target: **Churn (Yes/No)**

---

## ⚙️ Approach
- Data cleaning & preprocessing  
- Feature engineering  
- Model: Random Forest Classifier  
- Evaluation using Precision, Recall, F1-score  

---

## 🚀 API

Endpoint: `POST /predict`

### Input
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.85
}

{
  "churn_probability": 0.47,
  "churn_risk": "Medium"
}

pip install pandas numpy scikit-learn fastapi uvicorn shap
python -m src.train_model
uvicorn api.main:app --reload



📈 Insights
Month-to-month contracts have higher churn
High monthly charges increase churn risk
Low tenure customers are more likely to churn

🛠️ Tech Stack

Python, Pandas, Scikit-learn, SHAP, FastAPI

