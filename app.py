from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import numpy as np
import io
from PIL import Image
import tensorflow as tf
import joblib
import json
import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import traceback

# Verify TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Initialize the FastAPI app
app = FastAPI(
    title="AQI Classification API",
    description="API to predict AQI class and water quality using hybrid and regression models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Allow requests from Angular app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AQI model and components
try:
    with open('aqi_model_package/metadata.json', 'r') as f:
        metadata = json.load(f)
    class_names = metadata['class_names']
    numerical_cols = metadata['numerical_cols']
    scaler = joblib.load('aqi_model_package/scaler.pkl')
    model = tf.keras.models.load_model('aqi_model_package/model.keras')
except Exception as e:
    raise Exception(f"Failed to load AQI model or components: {str(e)}")

# Load water quality model and scaler
try:
    wq_model = joblib.load('wq_model_package/best_wqi_model.pkl')
    wq_scaler = joblib.load('wq_model_package/scaler.pkl')
    print("Water quality scaler feature names:", wq_scaler.feature_names_in_)
except Exception as e:
    raise Exception(f"Failed to load water quality model or scaler: {str(e)}")

# Preprocessing functions
def preprocess_image(image: bytes, target_size=(224, 224)):
    try:
        img = Image.open(io.BytesIO(image))
        img = img.resize(target_size)
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def preprocess_numerical(data: list, numerical_cols: list, scaler):
    try:
        data_df = pd.DataFrame([data], columns=numerical_cols)
        data_scaled = scaler.transform(data_df)
        return data_scaled
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing numerical data: {str(e)}")

def preprocess_numerical_wq(data: dict, scaler):
    try:
        expected_features = scaler.feature_names_in_  # Get features from scaler
        data_df = pd.DataFrame([data])[expected_features]  # Ensure order matches training
        print("Water quality input features:", data_df.columns.tolist())
        data_scaled = scaler.transform(data_df)
        return data_scaled
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing water quality data: {str(e)}")

# AQI prediction endpoint
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    pm25: float = 0.0,
    pm10: float = 0.0,
    o3: float = 0.0,
    co: float = 0.0,
    so2: float = 0.0,
    no2: float = 0.0,
    year: int = datetime.now().year,
    month: int = datetime.now().month,
    day: int = datetime.now().day,
    hour: int = datetime.now().hour
):
    try:
        print(f"Received AQI values: pm25={pm25}, pm10={pm10}, o3={o3}, co={co}, so2={so2}, no2={no2}, year={year}, month={month}, day={day}, hour={hour}")

        image_data = await image.read()
        image_processed = preprocess_image(image_data)

        numerical_data = [pm25, pm10, o3, co, so2, no2, year, month, day, hour]
        numerical_processed = preprocess_numerical(numerical_data, numerical_cols, scaler)

        prediction = model.predict([image_processed, numerical_processed])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]
        probabilities = prediction[0].tolist()

        return JSONResponse(content={
            "predicted_class": predicted_label,
            "probabilities": probabilities,
            "input_numerical_data": {
                "pm25": pm25, "pm10": pm10, "o3": o3, "co": co, "so2": so2, "no2": no2,
                "year": year, "month": month, "day": day, "hour": hour
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AQI prediction error: {str(e)}")

# Water quality prediction endpoint
@app.post("/predict/water")
async def predict_water(
    temp: float = 25.0,
    do: float = 0.0,
    ph: float = 0.0,
    conductivity: float = 0.0,
    bod: float = 0.0,
    nitrate_nitrite: float = 0.0,
    total_coliform: float = 0.0,
    year: int = datetime.now().year
):
    try:
        print(f"Received water quality values: temp={temp}, do={do}, ph={ph}, conductivity={conductivity}, bod={bod}, nitrate_nitrite={nitrate_nitrite}, total_coliform={total_coliform}, year={year}")

        # Input validation
        if ph < 0 or ph > 14:
            raise HTTPException(status_code=400, detail="pH must be between 0 and 14")
        if temp < -10 or temp > 50:
            raise HTTPException(status_code=400, detail="Temperature must be between -10°C and 50°C")
        if do < 0 or do > 20:
            raise HTTPException(status_code=400, detail="Dissolved Oxygen must be between 0 and 20 mg/L")
        if conductivity < 0 or conductivity > 1000:
            raise HTTPException(status_code=400, detail="Conductivity must be between 0 and 1000 µmhos/cm")
        if bod < 0 or bod > 200:
            raise HTTPException(status_code=400, detail="B.O.D. must be between 0 and 200 mg/L")
        if nitrate_nitrite < 0 or nitrate_nitrite > 500:
            raise HTTPException(status_code=400, detail="Nitrate + Nitrite must be between 0 and 500 mg/L")
        if total_coliform < 0 or total_coliform > 100000:
            raise HTTPException(status_code=400, detail="Total Coliform must be between 0 and 100,000 MPN/100ml")

        # Prepare numerical data
        numerical_data = {
            'temp': temp,
            'do': do,
            'ph': ph,
            'conductivity': conductivity,
            'bod': bod,
            'nitrate_nitrite': nitrate_nitrite,
            'total_coliform': total_coliform,
            'year': year
        }

        # Preprocess and predict
        numerical_processed = preprocess_numerical_wq(numerical_data, wq_scaler)
        wqi_predicted = wq_model.predict(numerical_processed)[0]

        # Classify quality
        if wqi_predicted <= 25:
            quality_class = "Excellent"
            purity = "Pure"
        elif wqi_predicted <= 50:
            quality_class = "Good"
            purity = "Pure"
        elif wqi_predicted <= 75:
            quality_class = "Poor"
            purity = "Not Pure"
        elif wqi_predicted <= 100:
            quality_class = "Very Poor"
            purity = "Not Pure"
        else:
            quality_class = "Unsuitable"
            purity = "Not Pure"

        return JSONResponse(content={
            "wqi": float(wqi_predicted),
            "quality_class": quality_class,
            "purity": purity,
            "input_numerical_data": numerical_data
        })
    except Exception as e:
        print("Error during water quality prediction:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Water quality prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)