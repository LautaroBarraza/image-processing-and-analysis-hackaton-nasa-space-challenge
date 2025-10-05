import os
import time
import warnings
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tslearn.clustering import TimeSeriesKMeans
import joblib

warnings.filterwarnings('ignore')

class TimeSeriesClassifier:
    """
    Encapsula el proceso de carga, entrenamiento, predicción y visualización
    de un modelo de clustering para series temporales.
    """
    def __init__(self, config):
        """Inicializa el clasificador con los parámetros de configuración."""
        self.config = config
        self.model = None
        self.full_timeseries = None
        self.labels = None
        self.results_map = None

        # Crear directorios de salida si no existen
        os.makedirs(self.config.OUTPUT_FOLDER_PLOTS, exist_ok=True)
        os.makedirs(self.config.OUTPUT_FOLDER_MODEL, exist_ok=True)
        print(f"Los gráficos se guardarán en: {os.path.abspath(self.config.OUTPUT_FOLDER_PLOTS)}")
        print(f"El modelo se guardará en: {os.path.abspath(self.config.OUTPUT_FOLDER_MODEL)}")
    
    

    def _load_and_sample_data(self):
        """
        Carga la máscara de calidad y luego carga solo los datos de series
        temporales que cumplen con el criterio de calidad.
        """
        print("\\n--- PASO 1: Cargando datos de alta calidad usando la máscara ---")
        
        if not os.path.exists(self.config.QUALITY_MASK_PATH):
            raise FileNotFoundError(f"No se encontró la máscara de calidad en {self.config.QUALITY_MASK_PATH}. "
                                    "Por favor, ejecuta 'run_preprocessing.py' primero.")
        quality_mask = np.load(self.config.QUALITY_MASK_PATH).flatten()
        
        with h5py.File(self.config.DATASET_PATH, 'r') as hf:
            full_timeseries_raw = hf['timeseries'][:]
        
        self.full_timeseries = full_timeseries_raw[quality_mask]
        
        print(f"✓ Dataset filtrado por calidad. Forma final: {self.full_timeseries.shape}")
        return self.full_timeseries

    def train(self):
        """Entrena el modelo TimeSeriesKMeans con una muestra de los datos."""
        if self.full_timeseries is None:
            self._load_and_sample_data()

        # Tomar muestra para entrenamiento
        n_series_total = self.full_timeseries.shape[0]
        random_indices = np.random.choice(n_series_total, self.config.SAMPLE_SIZE, replace=False)
        X_train = self.full_timeseries[random_indices]
        X_train = np.nan_to_num(X_train)
        print(f"✓ Muestra de {self.config.SAMPLE_SIZE} series lista para el entrenamiento.")

        print("\n--- PASO 2: Ejecutando el clustering ---")
        print("   (Esto puede tardar unos minutos)...")
        start_time = time.time()
        
        self.model = TimeSeriesKMeans(
            n_clusters=self.config.K_CLUSTERS,
            metric="euclidean",
            max_iter=50,
            n_init=3,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train)
        end_time = time.time()
        print(f"✓ Modelo entrenado en {end_time - start_time:.2f} segundos.")

    def predict(self):
        """Predice las etiquetas para todo el dataset en lotes."""
        if self.model is None:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir. Llama al método 'train()'.")
        
        print("\n--- PASO 3: Asignando patrones a cada píxel del mapa (por lotes) ---")
        start_time = time.time()
        full_timeseries_cleaned = np.nan_to_num(self.full_timeseries)
        n_series_total = full_timeseries_cleaned.shape[0]
        
        all_labels_list = []
        num_batches = int(np.ceil(n_series_total / self.config.BATCH_SIZE))

        for i in range(0, n_series_total, self.config.BATCH_SIZE):
            print(f"  -> Procesando lote {i // self.config.BATCH_SIZE + 1} de {num_batches}...")
            batch = full_timeseries_cleaned[i:i + self.config.BATCH_SIZE]
            batch_labels = self.model.predict(batch)
            all_labels_list.append(batch_labels)
        
        self.labels = np.concatenate(all_labels_list)
        self.results_map = self.labels.reshape(self.config.MAP_HEIGHT, self.config.MAP_WIDTH)
        end_time = time.time()
        print(f"✓ Mapa de comportamientos generado en {end_time - start_time:.2f} segundos.")

    def plot_results(self):
        """Visualiza y guarda los patrones y el mapa de clasificación."""
        if self.model is None or self.results_map is None:
            raise RuntimeError("El modelo debe ser entrenado y la predicción realizada antes de visualizar.")

        print("\n--- PASO 4: Visualizando y guardando los resultados ---")
        
        # Gráfico de patrones (centroides)
        plt.figure(figsize=(12, 8))
        for i in range(self.config.K_CLUSTERS):
            plt.subplot(2, 2, i + 1)
            plt.plot(self.model.cluster_centers_[i].ravel(), "r-")
            plt.title(f"Patrón/Cluster {i}")
            plt.xlabel("Tiempo (Número de Imagen)")
            plt.ylabel("Deformación (mm)")
        plt.tight_layout()
        plt.suptitle(f"Patrones de Comportamiento Encontrados (k={self.config.K_CLUSTERS})", fontsize=16, y=1.05)
        plt.savefig(os.path.join(self.config.OUTPUT_FOLDER_PLOTS, f"patrones_clusters_k{self.config.K_CLUSTERS}.png"), dpi=150)
        print("✓ Gráfico de patrones guardado.")
        plt.close()

        # Mapa de clasificación
        cmap = ListedColormap(['gray', 'blue', 'red', 'green'])
        labels_desc = [f'Patrón {i}' for i in range(self.config.K_CLUSTERS)]
        plt.figure(figsize=(14, 10))
        im = plt.imshow(self.results_map, cmap=cmap)
        plt.title(f'Mapa de Comportamiento de Deformación (k={self.config.K_CLUSTERS})', fontsize=16)
        cbar = plt.colorbar(im, ticks=range(self.config.K_CLUSTERS)); cbar.ax.set_yticklabels(labels_desc)
        plt.savefig(os.path.join(self.config.OUTPUT_FOLDER_PLOTS, f"mapa_clasificado_k{self.config.K_CLUSTERS}.png"), dpi=150)
        print("✓ Mapa de clasificación guardado.")
        plt.close()

    def save_results(self):
        """Guarda el modelo entrenado y las etiquetas para su uso posterior."""
        print("\n--- PASO 5: Guardando resultados para la interpretación ---")
        joblib.dump(self.model, os.path.join(self.config.OUTPUT_FOLDER_MODEL, 'kmeans_model.joblib'))
        np.save(os.path.join(self.config.OUTPUT_FOLDER_MODEL, 'all_labels.npy'), self.labels)
        print(f"✓ Modelo y etiquetas guardados en '{self.config.OUTPUT_FOLDER_MODEL}'.")

    def run_classification_pipeline(self):
        """Ejecuta el pipeline completo: cargar, entrenar, predecir y visualizar."""
        self._load_and_sample_data()
        self.train()
        self.predict()
        self.plot_results()
        self.save_results()
        print("\n\n🎯 ¡ANÁLISIS DE CLASIFICACIÓN FINALIZADO! 🎯")