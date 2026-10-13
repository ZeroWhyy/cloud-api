from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
n = 2500

u_left = rng.uniform(0.4, 1.2, n)
u_right = rng.uniform(0.4, 1.2, n)
speed = rng.normal(15, 4, n)
theta = rng.uniform(-np.pi, np.pi, n)

u_diff = u_left - u_right
z = 0.2 + 2.8 * u_diff + 0.08 * speed + 0.4 * np.sin(theta)
p_turn = 1 / (1 + np.exp(-z))
turn = rng.binomial(1, p_turn)

df = pd.DataFrame({
    "u_left": u_left,
    "u_right": u_right,
    "speed": speed,
    "theta": theta,
    "turn": turn,
})

X = df[["u_left", "u_right", "speed", "theta"]]
y = df["turn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"F1 en test: {f1_score(y_test, y_pred):.3f}")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
model_path = models_dir / "modelo_giro.joblib"
joblib.dump(model, model_path)
print(f"Modelo guardado en: {model_path}")