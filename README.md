---
title: fertilizer_pred
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
sdk_version: "3.0.0"
app_file: app.py
pinned: false
app_port: 8000
---

# Fertilizer Prediction Model API

This is a FastAPI-based web service that predicts the appropriate fertilizer based on soil parameters.

## Features

- Predicts fertilizer type based on soil composition
- Handles topsoil, subsoil, and deepsoil parameters
- Considers soil type and crop type
- Returns both class index, fertilizer name, and prediction probabilities

## Input Parameters

Each soil layer (topsoil, subsoil, deepsoil) contains 6 parameters:
- Temperature (°C)
- Humidity (%)
- pH
- Nitrogen (N) content
- Phosphorus (P) content
- Potassium (K) content

Additional parameters:
- Soil type (integer: 0-5)
- Crop type (integer: 0 for rice, 1 for coconut)

## API Endpoint

- **POST** `/predict`
  - Input format:
    ```json
    {
        "topsoil": [temperature, humidity, pH, N, P, K],
        "subsoil": [temperature, humidity, pH, N, P, K],
        "deepsoil": [temperature, humidity, pH, N, P, K],
        "soil_type": int,
        "crop_type": int
    }
    ```
  - Response format:
    ```json
    {
        "predicted_class": int,
        "fertilizer": string,
        "probabilities": {
            "DAP and MOP": float,
            "Good NPK": float,
            "MOP": float,
            "Urea and DAP": float,
            "Urea and MOP": float,
            "Urea": float,
            "DAP": float
        }
    }
    ```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/fertilizer-prediction.git
cd fertilizer-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Usage Example

```python
import requests

API_URL = "http://localhost:8000/predict"

input_data = {
    "topsoil": [25.3, 55.2, 6.5, 90, 40, 60],
    "subsoil": [14.1, 60.0, 6.4, 85, 38, 58],
    "deepsoil": [23.5, 12.0, 5.3, 0, 0, 23],
    "soil_type": 2,
    "crop_type": 1
}

response = requests.post(API_URL, json=input_data)
print(response.json())
```

## Docker Deployment

Build and run the Docker container:
```bash
docker build -t fertilizer-prediction .
docker run -p 8000:8000 fertilizer-prediction
``` 