"""
schemas.py
Define los esquemas de entrada y salida de la API usando Pydantic.

Pydantic valida automáticamente que los datos recibidos tengan
el tipo y formato correcto, devolviendo un error 422 claro si no.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PassengerInput(BaseModel):
    """
    Representa los datos de un pasajero del Titanic para hacer una predicción.

    Debe coincidir con las features usadas en el entrenamiento:
    pclass, sex, age, sibsp, parch, fare, embarked.
    """

    pclass: int = Field(..., ge=1, le=3, description="Clase del pasajero (1, 2 o 3)")
    sex: str = Field(..., description="Sexo del pasajero: 'male' o 'female'")
    age: float = Field(..., ge=0, le=100, description="Edad del pasajero")
    sibsp: int = Field(..., ge=0, description="Número de hermanos/cónyuges a bordo")
    parch: int = Field(..., ge=0, description="Número de padres/hijos a bordo")
    fare: float = Field(..., ge=0, description="Tarifa pagada")
    embarked: str = Field(..., description="Puerto de embarque: 'C', 'Q' o 'S'")

    class Config:
        json_schema_extra = {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S"
            }
        }


class BatchPassengerInput(BaseModel):
    """Representa una lista de pasajeros para predicción en batch."""

    passengers: List[PassengerInput]


class PredictionOutput(BaseModel):
    """Representa el resultado de una predicción individual."""

    survived: int = Field(..., description="0 = no sobrevivió, 1 = sobrevivió")
    probability: float = Field(..., description="Probabilidad de supervivencia")
    timestamp: Optional[datetime] = None


class BatchPredictionOutput(BaseModel):
    """Representa el resultado de una predicción en batch."""

    predictions: List[PredictionOutput]