import time
import statistics
import requests

URL = "http://127.0.0.1:8000/predict"

data = {
    "u_left": 1.2,
    "u_right": 0.4,
    "speed": 12.7,
    "theta": 0.65
}

latencies = []

for i in range(20):
    start = time.perf_counter()
    response = requests.post(URL, json=data)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)
    
    print(
        f"Solicitud {i+1:02d} | "
        f"Status: {response.status_code} | "
        f"Latencia: {latency_ms:.2f} ms"
    )

print("\nResumen")
print("------------------------")
print(f"Promedio: {statistics.mean(latencies):.2f} ms")
print(f"Mediana:  {statistics.median(latencies):.2f} ms")
print(f"Minimo:   {min(latencies):.2f} ms")
print(f"Maximo:   {max(latencies):.2f} ms")