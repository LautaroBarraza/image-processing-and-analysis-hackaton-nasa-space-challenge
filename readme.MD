# Pipeline de Análisis de Deformación Terrestre con IA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-activo-brightgreen.svg)

Este repositorio contiene el pipeline de análisis de imágenes satelitales desarrollado por el equipo **UNICEN Exactas** como parte de nuestra solución para el **Hackathon NASA Space Apps Challenge**. El proyecto se enfoca en la automatización del procesamiento de datos de deformación del terreno (InSAR) para identificar patrones de comportamiento mediante modelos de Machine Learning no supervisado y predecir tendencias a futuro.

## 🛰️ Introducción

El objetivo de este pipeline es procesar grandes volúmenes de datos satelitales de radar de apertura sintética (SAR) para monitorear deformaciones en la superficie terrestre, como hundimientos o levantamientos. El sistema automatiza desde la descarga de los datos crudos hasta la generación de predicciones, pasando por un riguroso control de calidad y un análisis de clustering para detectar zonas con comportamientos anómalos.

---

## ✨ Características Principales

-   **Adquisición Automatizada:** Descarga datos InSAR directamente desde el Alaska Satellite Facility (ASF) para cualquier región del mundo.
-   **Procesamiento de Datos:** Convierte eficientemente los archivos GeoTIFF crudos a un formato H5 optimizado para análisis.
-   **Control de Calidad:** Genera una máscara de coherencia para filtrar píxeles de baja calidad, asegurando la fiabilidad de los resultados.
-   **Clustering Inteligente:** Utiliza K-Means para series temporales (`tslearn`) para agrupar píxeles con patrones de deformación similares.
-   **Interpretación y Predicción:** Analiza la velocidad de deformación de cada clúster y utiliza el modelo Prophet de Meta para predecir su comportamiento a largo plazo.
-   **Alta Configurabilidad:** Diseñado para ser fácilmente adaptable a nuevas regiones de estudio modificando un único archivo de configuración.

---

## 📂 Estructura del Repositorio

El proyecto está organizado de manera modular para facilitar su mantenimiento y escalabilidad.

|-- 📂 data/                  # Carpeta para los datos (no incluida en el repo)
|   |-- 📂 insar_raw_downloads/ # Datos crudos descargados
|   |-- mendoza_timeseries_dataset.h5 # Dataset procesado
|
|-- 📂 output/                # Carpeta para los resultados generados
|   |-- 📂 analisis_clustering/ # Gráficos y reportes de la interpretación
|   |-- 📂 quality_mask/      # Máscara de calidad generada
|   |-- 📂 saved_model/       # Modelo K-Means entrenado y etiquetas
|
|-- 📂 src/                   # Código fuente del proyecto
|   |-- 📂 logic/             # Clases que encapsulan la lógica principal
|   |   |-- data_manager.py     # Lógica de descarga y procesamiento H5
|   |   |-- preprocessor.py     # Lógica del control de calidad
|   |   |-- optimizer.py        # Lógica del método del codo
|   |   |-- classifier.py       # Lógica del clustering K-Means
|   |   |-- interpreter.py      # Lógica de análisis y predicción
|   |
|   |-- config.py             # ¡El archivo más importante para la configuración!
|   |-- run_*.py              # Scripts ejecutables para cada paso del pipeline
|
|-- README.md                 # Este archivo
|-- requirements.txt          # Dependencias de Python

---

## 🔧 Cómo Adaptarlo a una Nueva Región

Este pipeline fue diseñado para ser flexible. Para analizar una nueva área geográfica, solo necesitas modificar el archivo `src/config.py`.

**Los parámetros clave a cambiar son:**

1.  **`ASF_SEARCH_PARAMS`**:
    -   `'intersectsWith'`: Este es el más importante. Debes reemplazar el polígono de Mendoza por el **polígono de tu nueva área de interés**. Puedes obtenerlo fácilmente desde la [interfaz de búsqueda de ASF](https://search.asf.alaska.edu/).

2.  **`DATASET_H5_PATH`**:
    -   Cambia el nombre del archivo de salida (ej: `"neuquen_timeseries_dataset.h5"`) para mantener la organización.

3.  **`MAP_HEIGHT` y `MAP_WIDTH`**:
    -   Estos valores dependen de las dimensiones de los archivos GeoTIFF de tu nueva región. Deberás actualizarlos después del primer procesamiento para que los mapas se generen correctamente. El script de preprocesamiento (`run_preprocessing.py`) te informará sobre las dimensiones comunes de tus imágenes.

4.  **Parámetros del Modelo (Opcional)**:
    -   Después de ejecutar el método del codo (`run_find_optimal_k.py`) para tu nueva región, actualiza `K_CLUSTERS` con el nuevo valor óptimo.
    -   Puedes ajustar `CLUSTERS_TO_UNIFY` para que la predicción se enfoque en los nuevos clústeres de interés.

---

## ⚙️ Guía de Ejecución del Pipeline

Sigue estos pasos en orden desde la terminal, ubicándote en la carpeta raíz del proyecto.

### 1. Preparación del Entorno
Primero, clona el repositorio y crea un entorno virtual (recomendado).

```bash
# Clona el repositorio
git clone <url-del-repositorio>
cd <nombre-del-repositorio>

# Crea y activa un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instala las dependencias
pip install -r requirements.txt


# Ejecuta los siguientes scripts en orden. Cada uno representa una fase del análisis.
# PASO 0: Descarga los datos crudos y los convierte a H5
python src/run_data_preparation.py

# PASO 1: Crea la máscara de coherencia para el control de calidad
python src/run_preprocessing.py

# PASO 2: Ejecuta el método del codo para encontrar el 'k' óptimo
# (Recuerda actualizar config.py con el 'k' resultante)
python src/run_find_optimal_k.py

# PASO 3: Entrena el modelo de clustering y clasifica los píxeles
python src/run_classification.py

# PASO 4: Interpreta los resultados y genera la predicción a futuro
python src/run_interpretation.py

Al finalizar, todos los resultados (gráficos, reportes, modelo y datos procesados) estarán disponibles en la carpeta output/.

📄 Licencia
Este proyecto se distribuye bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

👥 Autores
Equipo: UNICEN Exactas

Evento: Hackathon NASA Space Apps Challenge
