"""
models.py
Define las tablas de la base de datos usando el ORM de SQLAlchemy.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from app.database.database import Base


class PredictionRecord(Base):
    """
    Representa un registro de predicción guardado en la base de datos.

    Guarda tanto los datos de entrada del pasajero como el resultado
    de la predicción, para poder auditar o analizar el uso de la API
    más adelante.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    # Datos de entrada del pasajero
    pclass = Column(Integer)
    sex = Column(String)
    age = Column(Float)
    sibsp = Column(Integer)
    parch = Column(Integer)
    fare = Column(Float)
    embarked = Column(String)

    # Resultado de la predicción
    survived = Column(Integer)
    probability = Column(Float)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)