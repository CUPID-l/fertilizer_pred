from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn
from typing import List
import os

app = FastAPI()

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
num_features = 23
num_classes = 7

model = FTTransformer(num_features=num_features, num_classes=num_classes).to(device)

# Check if model file exists
model_path = 'fttransf_new.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please make sure the model file is in the correct directory.")

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
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
    topsoil: List[float]
    subsoil: List[float]
    deepsoil: List[float]
    soil_type: int
    crop_type: int

@app.post("/predict")
async def predict_fertilizer(input_data: SoilInput):
    try:
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
            predicted_class = torch.argmax(output, dim=1).item()
            
        return {
            "predicted_class": predicted_class,
            "fertilizer": fertilizer_labels[predicted_class]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 