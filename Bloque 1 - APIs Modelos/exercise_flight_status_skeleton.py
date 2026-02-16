"""
Esqueleto del ejercicio: Mini-API de clima de vuelo.

Completa los TODOs para que la API funcione como se describe en el enunciado.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Mini Flight Status API")


# TODO: Completa el modelo Pydantic con los campos requeridos y validaciones.
class FlightStatusRequest(BaseModel):
    # flight_id: str
    # distance: int entre 100 y 5000
    # bad_weather: bool
    pass



@app.post("/flight-status")
async def flight_status(data: FlightStatusRequest):
    """
    Calcula un indice de riesgo simple:

    risk = distance / 5000 + (1 si bad_weather es true, si no 0)

    Devuelve:
    - flight_id
    - risk_score (redondeado a 2 decimales)
    - status: "low" | "medium" | "high"
    """
    # TODO: Extrae los campos de data
    # TODO: Calcula el risk_score
    # TODO: Determina el status segun los umbrales
    # TODO: Devuelve el JSON de respuesta
    raise HTTPException(status_code=501, detail="Not implemented")


# TODO (opcional): Endpoint de salud

