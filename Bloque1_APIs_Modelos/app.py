"""
Aplicación FastAPI para predicción de retrasos en vuelos usando un modelo de Machine Learning.

Esta API permite hacer predicciones individuales y por lotes sobre si un vuelo se retrasará
basándose en la distancia del vuelo y las condiciones meteorológicas. Está diseñada con fines
didácticos para enseñar el uso de FastAPI en aplicaciones de ML.

Características principales:
- Endpoint /predict: Predicción individual de retraso.
- Endpoint /predict-batch: Predicciones por lotes.
- Endpoint /info: Información sobre la API.
- Endpoint /metrics: Métricas de uso (número de predicciones, tiempo de actividad).
- Endpoint /simulate-error: Simulación de errores para testing.

El modelo utilizado es un clasificador binario entrenado previamente y guardado en 'flight_delay_model.pkl'.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import joblib
import time



# ------------------------- Modelo de datos Pydantic -------------------------
# Definimos un modelo Pydantic para validar automáticamente las entradas.
# Esto permite que FastAPI genere documentación interactiva en /docs con campos editables.
# Usamos Field para añadir constraints de validación (rangos, etc.).
class FlightData(BaseModel):
    flight_id: str
    distance: int = Field(..., ge=100, le=5000, description="Distancia del vuelo en km (100-5000)")
    bad_weather: bool



# ------------------------- Cargar modelo entrenado -------------------------
# El modelo de Machine Learning se carga al iniciar la aplicación.
# Se utiliza joblib para deserializar el modelo guardado en un archivo pickle.
# Si no se puede cargar, se lanza un error crítico que detiene la aplicación.
try:
    model = joblib.load("flight_delay_model.pkl")
    print("Modelo cargado exitosamente.")
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")



# ------------------------- Inicializar FastAPI -------------------------
# Se crea una instancia de FastAPI con un título descriptivo.
# FastAPI es un framework moderno para construir APIs web con Python, basado en estándares como OpenAPI.
app = FastAPI(title="Flight Delay ML API - Didáctica Avanzada")

# Contadores y estado simple para métricas
# prediction_count: Número total de predicciones realizadas desde el inicio.
# start_time: Momento en que se inició la aplicación, para calcular el uptime.
prediction_count = 0
start_time = time.time()



# ------------------------- Endpoint predict -------------------------
# Endpoint POST para hacer una predicción individual de retraso en un vuelo.
#
# Método: POST
# Ruta: /predict
# Cuerpo de la solicitud: JSON con flight_id, distance, bad_weather.
#
# Proceso:
# 1. Pydantic valida automáticamente los datos según el modelo FlightData.
# 2. Prepara las características para el modelo: [distance, bad_weather (como int)].
# 3. Usa el modelo para predecir la probabilidad de retraso.
# 4. Determina si el vuelo se considera retrasado (probabilidad > 0.5).
# 5. Incrementa el contador de predicciones.
# 6. Retorna un JSON con flight_id, delay_probability y delayed.
#
# Respuesta exitosa (200):
# {
#   "flight_id": "ABC123",
#   "delay_probability": 0.75,
#   "delayed": true
# }
#
# Errores posibles:
# - 422: Datos inválidos (Pydantic validation error).
# - 500: Error interno (problema con el modelo o procesamiento).
@app.post("/predict")
# Decorador que registra un endpoint HTTP POST en FastAPI, asigna la ruta y el método,
# y además lo incluye en el esquema OpenAPI para que aparezca en /docs automáticamente.
# async declara una función asíncrona compatible con el event loop: si hubiera operaciones
# de I/O (red, disco), se podrían hacer sin bloquear otras peticiones concurrentes.
async def predict_delay(data: FlightData):
    # Usamos un contador global para acumular métricas simples entre peticiones.
    # Es un estado compartido a nivel de proceso: se reinicia al reiniciar la app
    # y no es compartido entre múltiples workers si se ejecutan varios procesos.
    global prediction_count
    try:
        # Pydantic ya validó automáticamente los tipos y constraints
        flight_id = data.flight_id
        distance = data.distance
        bad_weather = data.bad_weather

        # Preparar características para el modelo: lista de listas con distance y bad_weather convertido a int
        features = [[distance, int(bad_weather)]]
        # Predecir la probabilidad usando predict_proba (probabilidad de clase positiva, retraso)
        probability = float(model.predict_proba(features)[0][1])
        # Determinar si se considera retrasado basado en umbral 0.5
        delayed = probability > 0.5

        # Incrementar contador global de predicciones
        prediction_count += 1

        # Retornar respuesta con resultados
        return {
            "flight_id": flight_id,
            "delay_probability": probability,
            "delayed": delayed
        }

    except Exception as e:
        # Capturar cualquier error y retornar 500
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")



# ------------------------- Endpoint predict-batch -------------------------
# Endpoint POST para hacer predicciones por lotes de retrasos en vuelos.
#
# Método: POST
# Ruta: /predict-batch
# Cuerpo de la solicitud: Lista de JSON, cada uno con flight_id, distance, bad_weather.
#
# Proceso:
# 1. Recibe una lista de datos JSON.
# 2. Verifica que sea una lista.
# 3. Para cada vuelo en la lista:
#    a. Valida los datos (si falla, ignora ese vuelo).
#    b. Prepara características y predice.
#    c. Agrega resultado a la lista.
#    d. Incrementa contador.
# 4. Retorna JSON con lista de predicciones y conteo total.
#
# Respuesta exitosa (200):
# {
#   "predictions": [
#     {"flight_id": "ABC123", "delay_probability": 0.75, "delayed": true},
#     ...
#   ],
#   "count": 5
# }
#
# Notas:
# - Vuelos inválidos se ignoran silenciosamente (no se incluyen en resultados).
# - Útil para procesar múltiples vuelos en una sola solicitud, optimizando rendimiento.
@app.post("/predict-batch")
# Decorador POST: vincula esta función a la ruta /predict-batch como endpoint de la API.
# async permite manejar muchas solicitudes concurrentes sin bloquear el servidor.
async def predict_batch(data_list: List[FlightData]):
    # Contador global reutilizado para sumar todas las predicciones del lote.
    global prediction_count
    try:
        results = []
        for data in data_list:
            # Pydantic ya validó automáticamente
            flight_id = data.flight_id
            distance = data.distance
            bad_weather = data.bad_weather
            features = [[distance, int(bad_weather)]]
            probability = float(model.predict_proba(features)[0][1])
            delayed = probability > 0.5
            # Agregar resultado a la lista
            results.append({
                "flight_id": flight_id,
                "delay_probability": probability,
                "delayed": delayed
            })
            # Incrementar contador por cada predicción exitosa
            prediction_count += 1

        # Retornar resultados y conteo
        return {"predictions": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")



# ------------------------- Endpoint info -------------------------
# Endpoint GET para obtener información general sobre la API.
#
# Método: GET
# Ruta: /info
# Sin parámetros.
#
# Retorna metadatos de la API, incluyendo nombre, descripción, versión y lista de características disponibles.
# Útil para que los clientes conozcan las capacidades del servicio.
#
# Respuesta (200):
# {
#   "service": "Flight Delay ML API",
#   "description": "API de ejemplo para enseñar FastAPI y ML",
#   "version": "1.0",
#   "features": ["predict", "predict-batch", "metrics", "simulate-error"]
# }
@app.get("/info")
def info():
    return {
        "service": "Flight Delay ML API",
        "description": "API de ejemplo para enseñar FastAPI y ML",
        "version": "1.0",
        "features": ["predict", "predict-batch", "metrics", "simulate-error"]
    }



# ------------------------- Endpoint metrics -------------------------
# Endpoint GET para obtener métricas de uso de la API.
#
# Método: GET
# Ruta: /metrics
# Sin parámetros.
#
# Calcula y retorna:
# - total_predictions: Número total de predicciones realizadas desde el inicio.
# - uptime_seconds: Tiempo en segundos que la aplicación ha estado ejecutándose.
#
# Útil para monitoreo y debugging del servicio.
#
# Respuesta (200):
# {
#   "total_predictions": 42,
#   "uptime_seconds": 3600
# }
@app.get("/metrics")
def metrics():
    # Calcular uptime restando el tiempo actual al tiempo de inicio
    uptime = int(time.time() - start_time)
    return {
        "total_predictions": prediction_count,
        "uptime_seconds": uptime
    }



# ------------------------- Endpoint simulate-error -------------------------
# Endpoint POST para simular errores en la API, útil para testing y enseñanza.
#
# Método: POST
# Ruta: /simulate-error
# Cuerpo: JSON con "raise_error": true/false.
#
# Si raise_error es true, lanza una HTTPException con código 418 (I'm a teapot) y mensaje personalizado.
# De lo contrario, retorna un mensaje de éxito.
#
# Útil para:
# - Probar manejo de errores en clientes.
# - Enseñar sobre códigos de estado HTTP.
# - Simular fallos en entornos de desarrollo.
#
# Respuesta si raise_error=false (200):
# {"status": "ok", "message": "No se produjo error"}
#
# Error si raise_error=true (418):
# {"detail": "Este es un error simulado para enseñar manejo"}
# ------------------------- Modelo de datos para simular error -------------------------
class ErrorData(BaseModel):
    raise_error: bool

@app.post("/simulate-error")
async def simulate_error(data: ErrorData):
    if data.raise_error:
        # Lanzar error simulado con código 418 (I'm a teapot)
        raise HTTPException(status_code=418, detail="Este es un error simulado para enseñar manejo")
    # Retornar éxito si no se solicita error
    return {"status": "ok", "message": "No se produjo error"}
