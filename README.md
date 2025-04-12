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
- Returns both class index and fertilizer name

## API Endpoint

- **POST** `/predict`
  - Input format:
    ```json
    {
        "topsoil": [float, float, float, float, float, float, float],
        "subsoil": [float, float, float, float, float, float, float],
        "deepsoil": [float, float, float, float, float, float, float],
        "soil_type": int,
        "crop_type": int
    }
    ```
  - Response format:
    ```json
    {
        "predicted_class": int,
        "fertilizer": string
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
    "topsoil": [25.3, 55.2, 0.0, 6.5, 90, 40, 60],
    "subsoil": [14.1, 60.0, 0.0, 6.4, 85, 38, 58],
    "deepsoil": [23.5, 12.0, 0.0, 5.3, 0, 0, 23],
    "soil_type": 2,
    "crop_type": 5
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