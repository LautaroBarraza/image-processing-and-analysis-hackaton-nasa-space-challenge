import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from scipy.signal import medfilt
import logging
import time

logging.getLogger('prophet').setLevel(logging.ERROR)

class ClusterInterpreter:
    """
    Carga los resultados de la clasificación e interpreta los clusters.
    """
    def __init__(self, config):
        self.config = config
        self.labels = None
        self.full_timeseries = None
        self._load_dependencies()

    def _load_dependencies(self):
        """Carga los datos y las etiquetas necesarias para la interpretación."""
        print("--- Cargando datos y etiquetas para la interpretación ---")
        
        if not os.path.exists(self.config.QUALITY_MASK_PATH):
            raise FileNotFoundError(f"No se encontró la máscara de calidad. Ejecuta 'run_preprocessing.py'.")
        quality_mask = np.load(self.config.QUALITY_MASK_PATH).flatten()

        with h5py.File(self.config.DATASET_PATH, 'r') as hf:
            full_timeseries_raw = hf['timeseries'][:]
        self.full_timeseries = full_timeseries_raw[quality_mask]

        labels_path = os.path.join(self.config.OUTPUT_FOLDER_MODEL, 'all_labels.npy')
        if not os.path.exists(labels_path):
            raise FileNotFoundError("No se encontraron las etiquetas. Ejecuta 'run_classification.py'.")
        self.labels = np.load(labels_path)
        
        print("✓ Datos y etiquetas de alta calidad cargados correctamente.")

    def analyze_and_plot_velocities(self):
        """Calcula y grafica la distribución de velocidades para cada cluster."""
        print("\n" + "="*60)
        print("INICIANDO ANÁLISIS DE VELOCIDAD POR CLUSTER")
        print("="*60)
        
        n_images = self.full_timeseries.shape[1]
        total_years = (n_images * self.config.DAYS_PER_IMAGE) / 365.0
        time_vector_years = np.linspace(0, total_years, n_images)

        plt.figure(figsize=(15, 10))
        for i in range(self.config.K_CLUSTERS):
            cluster_indices = np.where(self.labels == i)[0]
            if len(cluster_indices) == 0: continue

            sample_size = min(len(cluster_indices), 10000)
            sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
            cluster_timeseries = self.full_timeseries[sample_indices]

            velocities = [np.polyfit(time_vector_years, np.nan_to_num(ts), 1)[0] for ts in cluster_timeseries if not np.all(np.isnan(ts))]
            if not velocities: continue
                
            plt.subplot(2, 2, i + 1)
            plt.hist(velocities, bins=50, range=(-150, 150), color='skyblue', ec='black')
            plt.title(f"Velocidades del Patrón {i}")
            plt.xlabel("Velocidad (mm/año)")
            plt.ylabel("Nº de Píxeles (muestra)")
            mean_vel = np.mean(velocities)
            plt.axvline(mean_vel, color='r', linestyle='dashed', linewidth=2)
            plt.text(0.05, 0.9, f'Media: {mean_vel:.1f} mm/año', transform=plt.gca().transAxes, color='r')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Interpretación de los Patrones por Velocidad", fontsize=18)
        plt.savefig(os.path.join(self.config.OUTPUT_FOLDER_PLOTS, "interpretacion_velocidades.png"))
        print("\n✓ Gráfico de interpretación de velocidades guardado.")
        plt.close()

    def predict_unified_zone(self):
        """
        Unifica los clusters de interés, suaviza la señal y predice la
        deformación futura utilizando Prophet.
        """
        print("\n" + "="*60)
        print("INICIANDO PREDICCIÓN CON PROPHET PARA ZONA UNIFICADA")
        print("="*60)

        start_time = time.time()
        
        # 1. Crear máscara para unificar los clusters de interés
        clusters_to_unify = self.config.CLUSTERS_TO_UNIFY
        mask_unificada = np.isin(self.labels, clusters_to_unify)
        
        if not np.any(mask_unificada):
            print(f"✗ ADVERTENCIA: No se encontraron píxeles para los clusters {clusters_to_unify}. Saltando predicción.")
            return

        print(f"Se unificarán los clusters {clusters_to_unify} para el análisis.")
        print(f"  -> Total de píxeles en la zona unificada: {np.sum(mask_unificada)}")

        # 2. Calcular la serie promedio y suavizarla
        serie_promedio = np.nanmean(self.full_timeseries[mask_unificada], axis=0)
        serie_suavizada = medfilt(serie_promedio, kernel_size=self.config.MEDIAN_FILTER_KERNEL_SIZE)
        print("  -> Serie temporal promediada y suavizada con filtro de mediana.")
        
        # 3. Preparar datos para Prophet
        fechas = pd.to_datetime(pd.date_range(
            start=self.config.PREDICTION_START_DATE,
            periods=serie_suavizada.shape[0],
            freq=f'{self.config.DAYS_PER_IMAGE}D'
        ))
        df_prophet = pd.DataFrame({'ds': fechas, 'y': serie_suavizada})

        # 4. Entrenar modelo Prophet y predecir
        print("  -> Entrenando modelo Prophet...")
        modelo = Prophet(changepoint_prior_scale=0.05)
        modelo.fit(df_prophet)

        print(f"  -> Realizando predicción a largo plazo hasta el año {self.config.PREDICTION_END_YEAR}...")
        end_date_str = f'{self.config.PREDICTION_END_YEAR}-12-31'
        num_future_days = (pd.to_datetime(end_date_str) - fechas.max()).days
        futuro = modelo.make_future_dataframe(periods=num_future_days)
        prediccion = modelo.predict(futuro)
        
        # 5. Extraer y guardar resultados
        reporte_prediccion = {'Zona Analizada': f'Zona General (Patrones {", ".join(map(str, clusters_to_unify))})'}
        for year in self.config.PREDICTION_YEAR_INTERVALS:
            pred_mean = prediccion[prediccion['ds'].dt.year == year]['yhat'].mean()
            reporte_prediccion[f'Predicción {year} (mm)'] = pred_mean
        
        df_reporte = pd.DataFrame([reporte_prediccion])
        csv_path = os.path.join(self.config.OUTPUT_FOLDER_PLOTS, "prediccion_zona_unificada.csv")
        df_reporte.to_csv(csv_path, index=False)
        print(f"\n✓ Resultados de la predicción guardados en: {csv_path}")
        print(df_reporte)

        # 6. Graficar y guardar la figura
        fig = modelo.plot(prediccion, figsize=(15, 8))
        ax = fig.gca()
        ax.plot(df_prophet['ds'], df_prophet['y'], 'k.', label='Datos Históricos (Promedio Suavizado)')
        ax.set_title(f'Predicción de Deformación para la Zona General (Clusters {clusters_to_unify})', fontsize=18)
        ax.set_xlabel('Fecha', fontsize=14)
        ax.set_ylabel('Deformación (mm)', fontsize=14)
        plt.legend()
        
        plot_path = os.path.join(self.config.OUTPUT_FOLDER_PLOTS, "grafico_prediccion_prophet.png")
        fig.savefig(plot_path, dpi=150)
        print(f"✓ Gráfico de predicción guardado en: {plot_path}")
        plt.close(fig)

        end_time = time.time()
        print(f"\nAnálisis de predicción completado en {end_time - start_time:.2f} segundos.")