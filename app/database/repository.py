"""
repository.py
Implementa el patrón Repository: encapsula el acceso a la base de datos
para guardar registros de predicciones.
"""

from sqlalchemy.orm import Session
from app.database.models import PredictionRecord
from app.api.schemas import PassengerInput, PredictionOutput


class PredictionRepository:
    """
    Encapsula las operaciones de base de datos relacionadas con
    los registros de predicciones.

    Attributes:
        db (Session): sesión activa de SQLAlchemy.
    """

    def __init__(self, db: Session):
        """
        Args:
            db (Session): sesión de base de datos, inyectada por FastAPI
                a través de la dependencia get_db().
        """
        self.db = db

    def save(self, passenger: PassengerInput, prediction: PredictionOutput) -> PredictionRecord:
        """
        Guarda un registro de predicción en la base de datos.

        Args:
            passenger (PassengerInput): datos de entrada del pasajero.
            prediction (PredictionOutput): resultado de la predicción.

        Returns:
            PredictionRecord: el registro ya guardado, con su id asignado.
        """
        record = PredictionRecord(
            pclass=passenger.pclass,
            sex=passenger.sex,
            age=passenger.age,
            sibsp=passenger.sibsp,
            parch=passenger.parch,
            fare=passenger.fare,
            embarked=passenger.embarked,
            survived=prediction.survived,
            probability=prediction.probability,
        )

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        return record