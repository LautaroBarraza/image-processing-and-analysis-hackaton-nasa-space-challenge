import config
# Cambiamos la línea de importación
from logic.classifier import TimeSeriesClassifier

def main():
    """
    Función principal para ejecutar el pipeline de clasificación.
    """
    print("==============================================")
    print("= INICIANDO ANÁLISIS DE CLASIFICACIÓN (K-MEANS) =")
    print("==============================================")
    
    classifier = TimeSeriesClassifier(config)
    classifier.run_classification_pipeline()

if __name__ == "__main__":
    main()