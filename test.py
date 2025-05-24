import requests

url = "http://localhost:8000/predict"
# Add query parameters to the URL
params = {
    "pm25": 348.0,
    "pm10": 199.0,
    "o3": 25.0,
    "co": 67.0,
    "so2": 10.0,
    "no2": 107.0
}
# Only the image is sent as form data
with open("test-images/DEL_SEV_2023-02-17-13.00-1-38.jpg", "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, params=params, files=files)

print(response.json())