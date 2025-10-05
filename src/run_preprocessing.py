import config
from logic.preprocessor import DataPreprocessor

def main():
    """
    Función principal para ejecutar el control de calidad.
    """
    preprocessor = DataPreprocessor(config)
    preprocessor.create_quality_mask()

if __name__ == "__main__":
    main()