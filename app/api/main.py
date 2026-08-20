"""
main.py
Punto de entrada de la API. Define los endpoints y conecta las clases
de ModelLoader, PredictionService y PredictionRepository.

Para correr localmente:
    uvicorn app.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from app.api.model_loader import ModelLoader
from app.api.prediction_service import PredictionService
from app.api.schemas import (
    PassengerInput,
    BatchPassengerInput,
    PredictionOutput,
    BatchPredictionOutput,
)
from app.database.database import get_db, init_db
from app.database.repository import PredictionRepository

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API para predecir la supervivencia de pasajeros del Titanic",
    version="1.0.0",
)

# Se cargan el modelo y el servicio una sola vez, al arrancar la API.
model_loader = ModelLoader(model_dir="app/models")
prediction_service = PredictionService(model_loader=model_loader)


@app.on_event("startup")
def on_startup():
    """Crea las tablas de la base de datos al arrancar la API."""
    init_db()


@app.get("/health")
def health_check():
    """
    Endpoint de healthcheck. Confirma que la API está corriendo
    y que el modelo se cargó correctamente.
    """
    return {"status": "ok", "model_loaded": model_loader.get_model() is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput, db: Session = Depends(get_db)):
    """
    Genera una predicción para un único pasajero y la guarda en la base
    de datos.

    Args:
        passenger (PassengerInput): datos del pasajero en el body del request.
        db (Session): sesión de base de datos, inyectada automáticamente.

    Returns:
        PredictionOutput: predicción (0/1) y probabilidad.
    """
    try:
        result = prediction_service.predict(passenger)
        result.timestamp = datetime.utcnow()

        repository = PredictionRepository(db)
        repository.save(passenger, result)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchPassengerInput, db: Session = Depends(get_db)):
    """
    Genera predicciones para una lista de pasajeros y las guarda en la
    base de datos.

    Args:
        batch (BatchPassengerInput): lista de pasajeros en el body del request.
        db (Session): sesión de base de datos, inyectada automáticamente.

    Returns:
        BatchPredictionOutput: lista de predicciones, una por pasajero.
    """
    try:
        results = prediction_service.predict_batch(batch.passengers)
        repository = PredictionRepository(db)

        for passenger, r in zip(batch.passengers, results):
            r.timestamp = datetime.utcnow()
            repository.save(passenger, r)

        return BatchPredictionOutput(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))