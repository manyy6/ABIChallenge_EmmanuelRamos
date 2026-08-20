"""
model_loader.py
Encapsula la carga del modelo de ML y sus encoders asociados.

Esta clase sigue el patrón Singleton informal: se instancia una sola vez
al arrancar la API, evitando cargar el modelo en cada request (ineficiente).
"""

import joblib
import os


class ModelLoader:
    """
    Carga y expone el modelo entrenado de Titanic y sus encoders.

    Attributes:
        model: modelo de clasificación entrenado (RandomForestClassifier).
        encoders (dict): encoders usados para transformar variables
            categóricas ('sex', 'embarked') de la misma forma que en
            el entrenamiento.
    """

    def __init__(self, model_dir: str = "app/models"):
        """
        Inicializa el loader y carga los artefactos desde disco.

        Args:
            model_dir (str): carpeta donde están los archivos .pkl
                del modelo y los encoders.

        Raises:
            FileNotFoundError: si no encuentra los archivos esperados.
        """
        self.model_dir = model_dir
        self.model = None
        self.encoders = None
        self._load()

    def _load(self) -> None:
        """Carga el modelo y los encoders desde los archivos .pkl."""
        model_path = os.path.join(self.model_dir, "titanic_model.pkl")
        encoders_path = os.path.join(self.model_dir, "encoders.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"No se encontraron los encoders en {encoders_path}")

        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoders_path)

    def get_model(self):
        """Retorna el modelo entrenado."""
        return self.model

    def get_encoders(self) -> dict:
        """Retorna el diccionario de encoders."""
        return self.encoders