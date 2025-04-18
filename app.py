from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn
from typing import List
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fertilizer Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model architecture
class FeatureTokenizer(torch.nn.Module):
    def __init__(self, num_features, embed_dim):
        super(FeatureTokenizer, self).__init__()
        self.embed = torch.nn.Linear(num_features, embed_dim)
    
    def forward(self, x):
        return self.embed(x)

class FTTransformer(torch.nn.Module):
    def __init__(self, num_features, num_classes, embed_dim=64, num_heads=4, num_layers=2):
        super(FTTransformer, self).__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.tokenizer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = 20  # 6 (topsoil) + 6 (subsoil) + 6 (deepsoil) + 1 (soil) + 1 (crop)
num_classes = 7

model = FTTransformer(num_features=num_features, num_classes=num_classes).to(device)

# Check if model file exists
model_path = 'fttransf_new.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please make sure the model file is in the correct directory.")

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
    
    # Test the model with a sample input
    test_input = torch.randn(1, num_features).to(device)
    with torch.no_grad():
        test_output = model(test_input)
        logger.info(f"Model test output shape: {test_output.shape}")
        logger.info(f"Model test output: {test_output}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise Exception(f"Error loading model: {str(e)}")

# Define fertilizer labels
fertilizer_labels = {
    0: "DAP and MOP",
    1: "Good NPK",
    2: "MOP",
    3: "Urea and DAP",
    4: "Urea and MOP",
    5: "Urea",
    6: "DAP"
}

class SoilInput(BaseModel):
    topsoil: List[float]  # [Temperature, Humidity, pH, N, P, K]
    subsoil: List[float]  # [Temperature, Humidity, pH, N, P, K]
    deepsoil: List[float]  # [Temperature, Humidity, pH, N, P, K]
    soil_type: int
    crop_type: int

@app.get("/")
async def root():
    return {"message": "Fertilizer Prediction API is running"}

@app.post("/predict")
async def predict_fertilizer(input_data: SoilInput):
    try:
        logger.info(f"Received prediction request: {input_data}")
        
        # Validate input lengths
        if len(input_data.topsoil) != 6 or len(input_data.subsoil) != 6 or len(input_data.deepsoil) != 6:
            error_msg = "Each soil layer must have exactly 6 features: [Temperature, Humidity, pH, N, P, K]"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Validate soil type and crop type
        if input_data.soil_type < 0 or input_data.soil_type > 5:
            error_msg = "soil_type must be between 0 and 5"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Remove crop_type validation to match test.ipynb
        # if input_data.crop_type < 0 or input_data.crop_type > 1:
        #     error_msg = "crop_type must be 0 (rice) or 1 (coconut)"
        #     logger.error(error_msg)
        #     raise HTTPException(status_code=400, detail=error_msg)
        
        # Combine all inputs into a single array
        input_array = np.array(
            input_data.topsoil + 
            input_data.subsoil + 
            input_data.deepsoil + 
            [input_data.soil_type, input_data.crop_type],
            dtype=np.float32
        )
        
        # Convert to tensor and predict
        input_tensor = torch.tensor(input_array).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            
            # Log detailed prediction information
            logger.info(f"Raw model output: {output}")
            logger.info(f"Class probabilities: {probabilities}")
            logger.info(f"Predicted class: {predicted_class}")
            
        result = {
            "predicted_class": predicted_class,
            "fertilizer": fertilizer_labels[predicted_class],
            "probabilities": {
                label: float(prob) for label, prob in zip(fertilizer_labels.values(), probabilities[0].tolist())
            }
        }
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error during prediction: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 