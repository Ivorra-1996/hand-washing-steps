# train_model.py
from ultralytics import YOLO

# Configuración
DATA_YAML_PATH = "./data.yaml"  # Ruta al archivo dataset.yaml
MODEL_NAME = "yolov8n.pt"  # Modelo base preentrenado YOLO
EPOCHS = 50  # Número de épocas de entrenamiento
IMAGE_SIZE = 640  # Tamaño de la imagen para el modelo

def train_model():
    print("Cargando modelo YOLO...")
    model = YOLO(MODEL_NAME)

    print("Iniciando entrenamiento...")
    results = model.train(
        data=DATA_YAML_PATH,  # Ruta al archivo de configuración YAML
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=16,  # Puedes ajustar este valor según tu GPU
        device="cpu"  # Selecciona la GPU (0) o CPU (-1)
    )

    print("¡Entrenamiento completado!")
    print(f"Resultados guardados en: {results.save_dir}")

if __name__ == "__main__":
    train_model()