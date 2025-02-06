from fastapi import FastAPI
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import pandas as pd
from pydantic import BaseModel


# Load the entire pipeline (preprocessor + model)
pipeline = joblib.load("loan_model_pipeline.pkl")  # Load the saved pipeline

# Initialize FastAPI app
app = FastAPI()

# Mount the "static" folder (for CSS/JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML file when accessing the root URL
@app.get("/")
def serve_homepage():
    return FileResponse("/stat")  # Serve the UI

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
