from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the trained model pipeline
pipeline = joblib.load("loan_model_pipeline.pkl")

# Define input schema
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

# Create FastAPI app
app = FastAPI()

# Enable CORS (Allow frontend to access backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any domain (you can restrict it)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
def predict_loan_status(data: LoanApplication):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Make a prediction
        prediction = pipeline.predict(input_data)[0]

        return {"loan_status": prediction}

    except Exception as e:
        return {"error": str(e)}
