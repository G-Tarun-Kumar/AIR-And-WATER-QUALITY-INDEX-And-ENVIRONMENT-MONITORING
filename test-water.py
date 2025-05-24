import joblib
import pandas as pd

wq_model = joblib.load('best_water_quality_model.pkl')
wq_scaler = joblib.load('scaler.pkl')

data = {
    'temp': 29.8,
    'do': 5.7,
    'ph': 7.2,
    'conductivity': 189.0,
    'bod': 2.0,
    'nitrate_nitrite': 0.2,
    'fecal_coliform': 4953.0,
    'total_coliform': 8391.0
}
columns = ['temp', 'do', 'ph', 'conductivity', 'bod', 'nitrate_nitrite', 'fecal_coliform', 'total_coliform']
data_df = pd.DataFrame([data], columns=columns)
scaled_data = wq_scaler.transform(data_df)
prediction = wq_model.predict(scaled_data)
print(prediction)