"""
test_prediction_service.py
Tests básicos para validar el funcionamiento de PredictionService.
"""

import pytest
from app.api.model_loader import ModelLoader
from app.api.prediction_service import PredictionService
from app.api.schemas import PassengerInput


@pytest.fixture
def prediction_service():
    """Fixture que crea una instancia real de PredictionService para los tests."""
    model_loader = ModelLoader(model_dir="app/models")
    return PredictionService(model_loader=model_loader)


def test_predict_returns_valid_output(prediction_service):
    """Verifica que predict() devuelve una predicción con formato correcto."""
    passenger = PassengerInput(
        pclass=3,
        sex="male",
        age=22.0,
        sibsp=1,
        parch=0,
        fare=7.25,
        embarked="S",
    )

    result = prediction_service.predict(passenger)

    assert result.survived in [0, 1]
    assert 0.0 <= result.probability <= 1.0


def test_predict_batch_returns_same_length(prediction_service):
    """Verifica que predict_batch() devuelve una predicción por cada pasajero enviado."""
    passengers = [
        PassengerInput(
            pclass=1, sex="female", age=30.0, sibsp=0, parch=0, fare=100.0, embarked="C"
        ),
        PassengerInput(
            pclass=3, sex="male", age=25.0, sibsp=0, parch=0, fare=8.0, embarked="S"
        ),
    ]

    results = prediction_service.predict_batch(passengers)

    assert len(results) == len(passengers)
    for r in results:
        assert r.survived in [0, 1]
        assert 0.0 <= r.probability <= 1.0


def test_predict_female_first_class_higher_survival_probability(prediction_service):
    """
    Verifica una regla conocida del dataset Titanic: mujeres de primera clase
    tuvieron mayor probabilidad de supervivencia que hombres de tercera clase.
    """
    high_chance_passenger = PassengerInput(
        pclass=1, sex="female", age=30.0, sibsp=0, parch=0, fare=100.0, embarked="C"
    )
    low_chance_passenger = PassengerInput(
        pclass=3, sex="male", age=30.0, sibsp=0, parch=0, fare=8.0, embarked="S"
    )

    high_result = prediction_service.predict(high_chance_passenger)
    low_result = prediction_service.predict(low_chance_passenger)

    assert high_result.survived == 1
    assert low_result.survived == 0