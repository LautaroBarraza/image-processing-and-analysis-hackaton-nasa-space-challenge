import os
import glob
import numpy as np
import rasterio
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Gestiona los pasos de preprocesamiento, como el cálculo de la
    coherencia promedio y la creación de una máscara de calidad.
    """
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.OUTPUT_FOLDER_QUALITY, exist_ok=True)
        print(f"La máscara de calidad se guardará en: {os.path.abspath(self.config.OUTPUT_FOLDER_QUALITY)}")

    def create_quality_mask(self):
        """
        Calcula la coherencia promedio de todos los archivos _corr.tif,
        manejando posibles diferencias de tamaño entre ellos, y
        genera una máscara booleana basada en un umbral.
        """
        print("\n" + "="*60)
        print("INICIANDO PREPROCESAMIENTO: CÁLCULO DE MÁSCARA DE CALIDAD")
        print("="*60)

        # --- PASO 1: BÚSQUEDA DE ARCHIVOS DE COHERENCIA ---
        search_pattern = os.path.join(self.config.INSAR_DATA_PATH, "**", "*_corr.tif")
        coherence_files = sorted(glob.glob(search_pattern, recursive=True))

        if not coherence_files:
            raise FileNotFoundError(f"No se encontraron archivos '_corr.tif' en la ruta: {self.config.INSAR_DATA_PATH}")
        
        print(f"✓ Se encontraron {len(coherence_files)} archivos de coherencia.")

        # --- PASO 2: DETERMINAR DIMENSIONES MÍNIMAS COMUNES ---
        print("\n--- Verificando dimensiones de las imágenes para recorte ---")
        dims = []
        for f in coherence_files:
            with rasterio.open(f) as ds:
                dims.append((ds.height, ds.width))
        
        min_height = min([d[0] for d in dims])
        min_width = min([d[1] for d in dims])
        print(f"✓ Las imágenes se recortarán a un tamaño común de: {min_height} x {min_width}.")

        # --- PASO 3: CALCULAR COHERENCIA PROMEDIO ---
        print("\n--- Calculando mapa de coherencia promedio ---")
        coherence_sum = np.zeros((min_height, min_width), dtype=np.float32)
        
        for i, file_path in enumerate(coherence_files):
            with rasterio.open(file_path) as src:
                # Leer y recortar cada banda al tamaño mínimo común
                band_data_cropped = src.read(1)[:min_height, :min_width]
                coherence_sum += band_data_cropped
            print(f"  -> Procesado archivo {i + 1}/{len(coherence_files)}", end="\\r")

        mean_coherence = coherence_sum / len(coherence_files)
        print("\n✓ Mapa de coherencia promedio calculado.")

        # --- PASO 4: CREAR Y GUARDAR LA MÁSCARA DE CALIDAD ---
        print("\n--- Generando y guardando la máscara de calidad ---")
        coherence_mask = (mean_coherence >= self.config.COHERENCE_THRESHOLD)
        np.save(self.config.QUALITY_MASK_PATH, coherence_mask)
        
        num_good_pixels = np.sum(coherence_mask)
        total_pixels = coherence_mask.size
        percentage = (num_good_pixels / total_pixels) * 100
        
        print(f"✓ Máscara de calidad guardada en: {self.config.QUALITY_MASK_PATH}")
        print(f"   -> {num_good_pixels} de {total_pixels} píxeles ({percentage:.2f}%) cumplen el umbral de coherencia >= {self.config.COHERENCE_THRESHOLD}.")
        print("\n🎯 PREPROCESAMIENTO FINALIZADO 🎯")