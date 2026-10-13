import requests

URL = "http://127.0.0.1:8000/predict"

data = {
    "u_left": 1.2,
    "u_right": 0.4,
    "speed": 12.7,
    "theta": 0.65
}

response = requests.post(URL, json=data)

print("Status code:", response.status_code)

try:
    print("Respuesta:", response.json())
except requests.exceptions.JSONDecodeError:
    print("Respuesta no JSON:", response.text)