# Entrenamiento de modelo
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Generar datos sintéticos
np.random.seed(0)
X = []
y = []

for _ in range(300):
    distance = np.random.randint(200, 3000)
    bad_weather = np.random.choice([0, 1])

    # Probabilidad de retraso
    delay_prob = 0.2
    if distance > 1500:
        delay_prob += 0.3
    if bad_weather == 1:
        delay_prob += 0.4

    delayed = np.random.rand() < delay_prob

    X.append([distance, bad_weather])
    y.append(int(delayed))

# Entrenar modelo
model = LogisticRegression()
model.fit(X, y)

# Guardar modelo para la API
joblib.dump(model, "model.pkl")
print("Modelo guardado en model.pkl")