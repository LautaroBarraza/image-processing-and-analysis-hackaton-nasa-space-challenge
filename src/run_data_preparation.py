import config
from logic.data_manager import DataDownloader, H5Processor

def main():
    """
    Ejecuta el pipeline completo de preparación de datos:
    1. Descarga los archivos de ASF.
    2. Procesa los archivos GeoTIFF en un único dataset H5.
    """
    print("==============================================")
    print("= INICIANDO PASO 0: PREPARACIÓN DE DATOS     =")
    print("==============================================")

    # Parte 1: Descargar y descomprimir
    downloader = DataDownloader(config)
    downloader.download_and_unzip()

    # Parte 2: Crear el archivo H5
    processor = H5Processor(config)
    processor.create_dataset()

    print("\\n🎯 PREPARACIÓN DE DATOS FINALIZADA 🎯")
    print("Ahora puedes proceder con el preprocesamiento y el análisis.")

if __name__ == "__main__":
    main()