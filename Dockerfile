# Imagen base ligera de Python
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /code

# Copiamos primero requirements.txt para aprovechar el cache de Docker:
# si no cambian las dependencias, Docker no las vuelve a instalar en cada build.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código del proyecto
COPY . .

# Entrenamos el modelo dentro de la imagen (para que el .pkl siempre exista)
RUN python app/models/train.py

# Puerto en el que corre la API dentro del contenedor
EXPOSE 8000

# Comando por defecto al iniciar el contenedor
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]