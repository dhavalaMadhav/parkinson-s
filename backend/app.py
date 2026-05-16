from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "parkinson_model.pkl")
model = joblib.load(model_path)

@app.get("/")
def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    return FileResponse(index_path)

class PredictRequest(BaseModel):
    data: list[float]

@app.post("/predict")
def predict(body: PredictRequest):
    values = [float(x) for x in body.data]
    input_array = np.array(values).reshape(1, -1)
    prediction = model.predict(input_array)
    result = "Parkinson Detected" if prediction[0] == 1 else "Healthy"
    return {"prediction": result}