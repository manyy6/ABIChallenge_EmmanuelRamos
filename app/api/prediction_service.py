"""
prediction_service.py
Contiene la lógica de negocio para transformar inputs y generar predicciones.
"""

import pandas as pd
from app.api.model_loader import ModelLoader
from app.api.schemas import PassengerInput, PredictionOutput


class PredictionService:
    """
    Servicio encargado de transformar los datos de entrada y generar
    predicciones usando el modelo cargado por ModelLoader.

    Attributes:
        model_loader (ModelLoader): instancia que expone el modelo y encoders.
    """

    def __init__(self, model_loader: ModelLoader):
        """
        Args:
            model_loader (ModelLoader): instancia ya inicializada con el
                modelo y encoders cargados.
        """
        self.model_loader = model_loader
        self.model = model_loader.get_model()
        self.encoders = model_loader.get_encoders()

    def _transform(self, passenger: PassengerInput) -> pd.DataFrame:
        """
        Transforma un PassengerInput en un DataFrame listo para el modelo,
        aplicando los mismos encoders usados en el entrenamiento.

        Args:
            passenger (PassengerInput): datos crudos del pasajero.

        Returns:
            pd.DataFrame: fila lista para pasar a model.predict().
        """
        sex_encoded = self.encoders["sex"].transform([passenger.sex])[0]
        embarked_encoded = self.encoders["embarked"].transform([passenger.embarked])[0]

        data = {
            "pclass": [passenger.pclass],
            "sex": [sex_encoded],
            "age": [passenger.age],
            "sibsp": [passenger.sibsp],
            "parch": [passenger.parch],
            "fare": [passenger.fare],
            "embarked": [embarked_encoded],
        }

        return pd.DataFrame(data)

    def predict(self, passenger: PassengerInput) -> PredictionOutput:
        """
        Genera una predicción individual para un pasajero.

        Args:
            passenger (PassengerInput): datos del pasajero.

        Returns:
            PredictionOutput: resultado con clase predicha y probabilidad.
        """
        X = self._transform(passenger)

        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][prediction])

        return PredictionOutput(survived=prediction, probability=probability)

    def predict_batch(self, passengers: list[PassengerInput]) -> list[PredictionOutput]:
        """
        Genera predicciones para una lista de pasajeros.

        Args:
            passengers (list[PassengerInput]): lista de pasajeros.

        Returns:
            list[PredictionOutput]: lista de resultados, uno por pasajero.
        """
        return [self.predict(p) for p in passengers]