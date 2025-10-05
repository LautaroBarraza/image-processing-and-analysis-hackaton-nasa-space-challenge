import os
import asf_search as asf # Necesario para los parámetros de búsqueda

# =================================================================================
# ARCHIVO DE CONFIGURACIÓN CENTRAL
# =================================================================================

# --- Rutas del Proyecto ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# --- (NUEVO) Parámetros de Adquisición y Procesamiento de Datos ---
# Para analizar una nueva región, modifica principalmente esta sección.

# 1. Directorio para guardar los ZIPs descargados y sus carpetas descomprimidas.
INSAR_RAW_DATA_PATH = os.path.join(DATA_DIR, "insar_raw_downloads")

# 2. Ruta final del archivo H5 que se creará. El resto del pipeline usará este archivo.
DATASET_H5_PATH = os.path.join(DATA_DIR, "mendoza_timeseries_dataset.h5")

# 3. Parámetros para la búsqueda en Alaska Satellite Facility (ASF)
#    Para encontrar el polígono de otra región, puedes usar: https://search.asf.alaska.edu/
ASF_SEARCH_PARAMS = {
    'platform': asf.PLATFORM.SENTINEL1,
    'processingLevel': asf.PRODUCT_TYPE.INTERFEROGRAM,
    'beamMode': asf.BEAMMODE.IW,
    'intersectsWith': 'POLYGON((-69.5 -33.5, -69.5 -32.5, -68.5 -32.5, -68.5 -33.5, -69.5 -33.5))' # Polígono para Mendoza
}

# --- Parámetros de Control de Calidad ---
# (IMPORTANTE) Esta ruta ahora apunta a la carpeta de datos crudos
# para que el preprocesador sepa dónde buscar los archivos _corr.tif
INSAR_QUALITY_DATA_PATH = INSAR_RAW_DATA_PATH
QUALITY_MASK_PATH = os.path.join(OUTPUT_DIR, "quality_mask", "quality_mask.npy")
COHERENCE_THRESHOLD = 0.4

# --- Parámetros del Mapa ---
# (Opcional) Deberás ajustar esto si las dimensiones de tu nueva región son muy diferentes
MAP_HEIGHT = 1493
MAP_WIDTH = 3337

# --- Parámetros del Optimizador (Método del Codo) ---
K_RANGE_TO_TEST = range(2, 8)
SAMPLE_SIZE_ELBOW = 20000

# --- Parámetros del Modelo de Clustering ---
K_CLUSTERS = 4
SAMPLE_SIZE = 50000
BATCH_SIZE = 100000

# --- Parámetros de Interpretación y Predicción ---
DAYS_PER_IMAGE = 12
PREDICTION_START_DATE = '2021-01-01'

# --- Parámetros de Predicción de Zona Unificada ---
CLUSTERS_TO_UNIFY = [1, 2]
PREDICTION_END_YEAR = 2050
PREDICTION_YEAR_INTERVALS = [2030, 2035, 2040, 2045, 2050]
MEDIAN_FILTER_KERNEL_SIZE = 3