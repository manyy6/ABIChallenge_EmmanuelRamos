"""
train.py
Script para entrenar un modelo simple de clasificación sobre el dataset del Titanic.

Objetivo: predecir si un pasajero sobrevivió (1) o no (0) según sus características.


"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def load_data() -> pd.DataFrame:
    """Carga el dataset del Titanic desde seaborn (mismo dataset que Kaggle)."""
    df = sns.load_dataset("titanic")
    return df


def preprocess(df: pd.DataFrame):
    """
    Limpieza y selección de features simples.

    Features usadas: pclass, sex, age, sibsp, parch, fare, embarked
    Target: survived
    """
    df = df.copy()

    # Seleccionamos columnas relevantes
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    target = "survived"

    df = df[features + [target]].dropna()

    # Codificamos variables categóricas
    le_sex = LabelEncoder()
    df["sex"] = le_sex.fit_transform(df["sex"])  # male=1, female=0

    le_embarked = LabelEncoder()
    df["embarked"] = le_embarked.fit_transform(df["embarked"])

    X = df[features]
    y = df[target]

    encoders = {"sex": le_sex, "embarked": le_embarked}

    return X, y, encoders


def train_model(X, y):
    """Entrena un RandomForestClassifier simple."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Accuracy en test: {accuracy:.4f}")

    return model


def save_model(model, encoders, output_dir="app/models"):
    """Guarda el modelo y los encoders entrenados como .pkl"""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, "titanic_model.pkl"))
    joblib.dump(encoders, os.path.join(output_dir, "encoders.pkl"))

    print(f"Modelo guardado en {output_dir}/titanic_model.pkl")
    print(f"Encoders guardados en {output_dir}/encoders.pkl")


if __name__ == "__main__":
    print("Cargando datos...")
    df = load_data()

    print("Preprocesando...")
    X, y, encoders = preprocess(df)

    print("Entrenando modelo...")
    model = train_model(X, y)

    print("Guardando modelo...")
    save_model(model, encoders)

    print("¡Listo!")