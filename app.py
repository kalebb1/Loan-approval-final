from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model pipeline
pipeline = joblib.load("loan_model_pipeline.pkl")

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")


# Define expected input structure
class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

app = FastAPI()

# Allow frontend (index.html) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")  # <-- Add this root route
def home():
    return {"message": "Loan Prediction API is running!"}

@app.post("/predict")
def predict_loan_status(data: LoanApplication):
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Use the pipeline to predict
        prediction = pipeline.predict(input_data)[0]

        return {"loan_status": prediction}

    except Exception as e:
        return {"error": str(e)}
