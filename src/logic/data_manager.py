import os
import glob
import zipfile
import requests
import asf_search as asf
import rasterio
import numpy as np
import h5py

class DataDownloader:
    """Gestiona la descarga y descompresión de datos desde ASF."""
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.INSAR_RAW_DATA_PATH, exist_ok=True)

    def download_and_unzip(self):
        print("\\n--- PASO 0.1: DESCARGANDO DATOS DE ALASKA SATELLITE FACILITY ---")
        results = asf.search(**self.config.ASF_SEARCH_PARAMS)
        print(f"Se encontraron {len(results)} archivos para descargar.")

        for i, product in enumerate(results):
            url = product.properties['url']
            filename = product.properties['fileName']
            zip_path = os.path.join(self.config.INSAR_RAW_DATA_PATH, filename)
            unzip_folder_path = zip_path.replace('.zip', '')

            print(f"\\n({i+1}/{len(results)}) Procesando: {filename}")
            if not os.path.exists(zip_path):
                print("  -> Descargando...")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except requests.RequestException as e:
                    print(f"  -> ERROR al descargar: {e}")
                    continue
            else:
                print("  -> El archivo ZIP ya existe. Saltando descarga.")

            if not os.path.exists(unzip_folder_path):
                print(f"  -> Descomprimiendo en: {unzip_folder_path}")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(unzip_folder_path)
                    print("  -> Descompresión completada.")
                except zipfile.BadZipFile:
                    print("  -> ERROR: El archivo no es un ZIP válido.")
            else:
                print("  -> La carpeta ya existe. Saltando descompresión.")

class H5Processor:
    """Procesa los archivos GeoTIFF descargados en un único archivo H5."""
    def __init__(self, config):
        self.config = config

    def create_dataset(self):
        print("\\n--- PASO 0.2: PROCESANDO ARCHIVOS GEO-TIFF A FORMATO H5 ---")
        search_pattern = os.path.join(self.config.INSAR_RAW_DATA_PATH, "**", "*_vert_disp.tif")
        disp_files = sorted(glob.glob(search_pattern, recursive=True))

        if not disp_files:
            raise FileNotFoundError("No se encontraron archivos _vert_disp.tif. Revisa la descarga.")
        print(f"Se encontraron {len(disp_files)} archivos de desplazamiento vertical.")

        dims = [(rasterio.open(f).height, rasterio.open(f).width) for f in disp_files]
        min_height = min(d[0] for d in dims)
        min_width = min(d[1] for d in dims)
        print(f"Las imágenes se recortarán a un tamaño común de: {min_height}x{min_width}.")

        num_files = len(disp_files)
        num_pixels = min_height * min_width
        
        with h5py.File(self.config.DATASET_H5_PATH, 'w') as hf:
            timeseries_dataset = hf.create_dataset('timeseries', (num_pixels, num_files), dtype='float32')
            print("Dataset H5 inicializado.")

            for i, file_path in enumerate(disp_files):
                with rasterio.open(file_path) as src:
                    disp_meters = src.read(1)[:min_height, :min_width]
                    disp_mm_flat = (disp_meters * 1000).flatten()
                    timeseries_dataset[:, i] = disp_mm_flat
                if (i + 1) % 20 == 0 or (i + 1) == num_files:
                    print(f"  -> Procesando y apilando archivo {i + 1}/{num_files}...")
        
        print(f"\\n✅ ¡Proceso finalizado! Dataset creado en: {self.config.DATASET_H5_PATH}")