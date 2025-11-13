# Predicción de Enfermedad Cardíaca con MLOps

## Descripción general

Este proyecto implementa un sistema de **aprendizaje automático** para la **predicción de enfermedad cardíaca**, dentro de una arquitectura **MLOps** moderna.  
Integra un modelo de clasificación entrenado con técnicas supervisadas, una **API REST** desarrollada con **FastAPI**, y despliegues automatizados mediante **Docker Compose** y **Kubernetes**.  
El flujo de trabajo está validado con **integración continua (CI)** mediante **GitHub Actions**, asegurando calidad de código y reproducibilidad.

## Estructura del proyecto

```text
MINIPROYECTO_S10/
│
├── app/
│ ├── api.py # Servicio FastAPI para predicciones
│ ├── feature_order.json # Orden de las características de entrada
│ └── model.joblib # Modelo entrenado serializado
│
├── data/
│ ├── raw/ # Datos originales
│ ├── cleaned/ # Datos procesados
│ └── eda_preprocessing.ipynb # Análisis exploratorio y preprocesamiento
│
├── notebooks/
│ └── model_training.ipynb # Entrenamiento del modelo
│
├── tests/
│ ├── init.py
│ ├── test_api.py # Pruebas del servicio FastAPI
│ └── test_model.py # Verificación del modelo y features
│
├── docker/
│ ├── Dockerfile # Contenedor de la API
│ ├── .dockerignore
│ └── requirements.txt # Dependencias del entorno
│
├── k8s/
│ ├── deployment.yaml # Definición de despliegue en Kubernetes
│ └── service.yaml # Servicio para exponer la API
│
├── .github/workflows/
│ └── ci.yml # Pipeline de integración continua (CI)
│
├── docker-compose.yml # Orquestación local con Docker Compose
├── .flake8 # Configuración de estilo
├── .gitignore # Archivos ignorados por Git
└── README.md # Documentación del proyecto
```

## Instalación y ejecución local

### Clonar el repositorio

```bash
git clone https://github.com/<tu_usuario>/MINIPROYECTO_S10.git
cd MINIPROYECTO_S10
```

### Instalar dependencias

```bash
pip install -r docker/requirements.txt
```

### Ejecutar el servicio FastAPI localmente

```bash
uvicorn app.api:app --reload
```

Acceda al servicio en [servicio](http://localhost:8000/docs)

Aquí encontrará la documentación interactiva Swagger UI para probar el endpoint /predict.

## Ejemplo de solicitud

```bash
{
    "features":[57, "Male", "ATA", 140, 240, 0, "Normal", 160, "N", 1.0, "Up"]
    }
```

El servicio responde con un objeto JSON que incluye:

- probabilidad_prediccion → Valor entre 0 y 1 que representa el riesgo estimado.
- prediccion → Resultado binario (1 = posible enfermedad, 0 = sin enfermedad).

## Despliegue con Docker Compose

### Construir la imagen

Desde la carpeta raíz del proyecto:

```bash
docker compose -f docker/docker-compose.yml build
```

### Ejecutar el contenedor

```bash
docker compose up
```

Acceda al servicio en [servicio](http://localhost:8000/docs)

### Detener los contenedores

```bash
docker compose down
```

## Despliegue con Kubernetes

### Aplicar los manifiestos

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Exponer el servico localmente

```bash
kubectl port-forward service/heart-service 8000:80
```

Acceda al servicio en [servicio](http://localhost:8000/docs)

## Integración continua con (CI/CD)

El flujo de CI/CD se ejecuta automáticamente en cada push al repositorio mediante GitHub Actions, con el archivo .github/workflows/ci.yml.

Etapas del pipeline

1. Configuración del entorno
    - Ubuntu-latest con Python 3.10
2. Instalación de dependencias
    - Instale las librerías listadas en docker/requirements.txt
3. Análisis estático
    - Ejecuta `flake8` sobre el directorio `app/`

4. Ejecución de pruebas

    - Ejecuta `pytest` sobre el directorio `tests/`

Esto garantiza la calidad del código, la reproducibilidad del entorno y la estabilidad del servicio antes de cualquier despliegue.

## Pruebas unitarias

Para ejecutar las pruebas localmente:

```bash
pytest -q
```

### Cobertura de pruebas  

- test_api.py
  - Verifica el endpoint /health
  - Valida la respuesta del endpoint raíz /
  - Evalúa el comportamiento del endpoint /predict (200, 400, 422, 500, 503)

- test_model.py
  - Comprueba la existencia del modelo model.joblib y del archivo feature_order.json
  - Verifica que el modelo puede generar probabilidades válidas

## Autoría y licencia

Proyecto Integrador de Aprendizaje Automático desarrollado como parte del curso de Machine Learning de la Universidad del Norte (Colombia). El contenido puede ser reutilizado con fines educativos o de investigación, citando la fuente correspondiente.

## Arquitectura MLOps

```text
+-----------------------------+
|       Datos (CSV)           |
|  Limpieza / EDA / Pandas    |
+-------------+---------------+
              |
              ▼
+-----------------------------+
|  Entrenamiento del modelo   |
|  Scikit-Learn / Pandas      |
|  Serialización (joblib)     |
+-------------+---------------+
              |
              ▼
+-----------------------------+
|      API REST - FastAPI     |
|  Carga del modelo y predice |
|  feature_order.json incluido|
+-------------+---------------+
              |
              ▼
+-----------------------------+
|    Contenerización Docker   |
|  Dockerfile + docker-compose|
+-------------+---------------+
              |
              ▼
+-----------------------------+
|  Orquestación Kubernetes    |
|  Deployment + Service (YAML)|
+-------------+---------------+
              |
              ▼
+-----------------------------+
| Integración Continua (CI/CD)|
| GitHub Actions + Pytest     |
| Flake8 + Validación de API  |
+-----------------------------+

```