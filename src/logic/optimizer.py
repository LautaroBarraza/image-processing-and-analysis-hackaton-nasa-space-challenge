import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

class OptimalKAnalyzer:
    """
    Encapsula la lógica para encontrar el número óptimo de clusters (k)
    utilizando el método del codo sobre los datos de alta calidad.
    """
    def __init__(self, config):
        self.config = config
        self.X_train = None
        self._prepare_data()

    def _prepare_data(self):
        """Carga la máscara de calidad y prepara una muestra de datos."""
        print("--- PASO 1: Preparando muestra de alta calidad para el análisis del codo ---")
        
        if not os.path.exists(self.config.QUALITY_MASK_PATH):
            raise FileNotFoundError(f"No se encontró la máscara de calidad en {self.config.QUALITY_MASK_PATH}. "
                                    "Por favor, ejecuta 'run_preprocessing.py' primero.")
        quality_mask = np.load(self.config.QUALITY_MASK_PATH).flatten()

        with h5py.File(self.config.DATASET_PATH, 'r') as hf:
            full_timeseries = hf['timeseries'][:]
        
        high_quality_series = full_timeseries[quality_mask]
        
        n_series_total = high_quality_series.shape[0]
        sample_size = min(n_series_total, self.config.SAMPLE_SIZE_ELBOW)
        random_indices = np.random.choice(n_series_total, sample_size, replace=False)
        self.X_train = np.nan_to_num(high_quality_series[random_indices])
        
        print(f"✓ Muestra de {sample_size} series de alta calidad lista.")

    def run_and_plot_elbow_method(self):
        """Ejecuta K-Means para un rango de 'k' y grafica la curva del codo."""
        print("\\n--- PASO 2: Ejecutando el Método del Codo ---")
        inertias = []
        k_range = self.config.K_RANGE_TO_TEST

        for k in k_range:
            print(f"  -> Entrenando modelo con k={k}...")
            start_time = time.time()
            model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=50, n_init=2, random_state=42, n_jobs=-1)
            model.fit(self.X_train)
            inertias.append(model.inertia_)
            end_time = time.time()
            print(f"     ✓ Completado en {end_time - start_time:.2f}s. Inercia: {model.inertia_:.2f}")

        print("\\n--- PASO 3: Visualizando la Curva del Codo ---")
        plt.figure(figsize=(12, 7))
        plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo para Encontrar el k Óptimo')
        plt.xticks(list(k_range))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_path = os.path.join(self.config.OUTPUT_FOLDER_PLOTS, "metodo_del_codo.png")
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Gráfico guardado en: {os.path.abspath(plot_path)}")
        plt.show()