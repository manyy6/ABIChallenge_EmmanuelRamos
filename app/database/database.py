"""
database.py
Configura la conexión a la base de datos usando SQLAlchemy.

Por defecto usa SQLite (un archivo local, sin necesidad de instalar
un servidor de base de datos), ideal para correr el proyecto localmente
o dentro de Docker.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# URL de conexión. Se puede sobreescribir con una variable de entorno
# DATABASE_URL para usar Postgres, MySQL, etc. en producción.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

# connect_args solo es necesario para SQLite (permite uso multi-hilo)
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    Generador de sesiones de base de datos.

    Se usa como dependencia en FastAPI: abre una sesión, la entrega,
    y la cierra automáticamente al terminar el request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Crea las tablas en la base de datos si no existen."""
    Base.metadata.create_all(bind=engine)