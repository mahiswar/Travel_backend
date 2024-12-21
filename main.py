from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model_path = "xgboost_smote_model.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    raise Exception(f"Error loading the model: {e}")

# Define the input schema using Pydantic
class PredictionInput(BaseModel):
    Age: int
    AnnualIncome: float
    FamilyMembers: int
    ChronicDiseases: int
    EmploymentType: str
    GraduateStatus: int
    FrequentFlyer: int
    TravelAbroad: int

# Encode categorical variables as required by your model
def preprocess_input(data: PredictionInput):
    employment_type_mapping = {"Salaried": 0, "Self-Employed": 1}
    if data.EmploymentType not in employment_type_mapping:
        raise HTTPException(status_code=400, detail="Invalid EmploymentType")
    
    # Map EmploymentType to numerical
    employment_type_encoded = employment_type_mapping[data.EmploymentType]
    
    # Create a numpy array for the model
    features = np.array([
        data.Age,
        data.AnnualIncome,
        data.FamilyMembers,
        data.ChronicDiseases,
        employment_type_encoded,
        data.GraduateStatus,
        data.FrequentFlyer,
        data.TravelAbroad
    ]).reshape(1, -1)
    
    return features

@app.post("/predict")
async def predict_travel_insurance(input_data: PredictionInput):
    try:
        # Preprocess the input data
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # Return the result
        return {
            "prediction": int(prediction[0]),
            "description": "1 indicates likely to purchase travel insurance, 0 indicates unlikely."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
