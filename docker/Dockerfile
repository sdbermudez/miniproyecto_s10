# docker/Dockerfile
FROM python:3.10-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar e instalar dependencias
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación
COPY app/ ./app/

# Exponer el puerto
EXPOSE 8000

# Comando por defecto para ejecutar FastAPI
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
