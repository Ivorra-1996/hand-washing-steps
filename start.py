import cv2
from ultralytics import YOLO
import threading
import time

# Constantes de configuración
MODEL_PATH = './runs/detect/train/weights/best.pt'
CAMERA_URL = 'http://192.168.0.100:4747/video'
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
FRAME_SKIP = 4  # Saltar frames para mejorar rendimiento
DEFAULT_VALUE = 0.25  # Valor por defecto para propiedades de la cámara
INFERENCE_INTERVAL = 0.1  # Intervalo de tiempo entre inferencias (en segundos)

# Variables globales
frame = None
ret = False
stop_thread = False


def configure_camera(cap):
    """Configura los parámetros de la cámara."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, FRAME_SKIP)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Configuración de propiedades comunes
    properties = [
        cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_EXPOSURE,
        cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_CONTRAST, cv2.CAP_PROP_SATURATION,
        cv2.CAP_PROP_HUE, cv2.CAP_PROP_GAIN, cv2.CAP_PROP_SHARPNESS, cv2.CAP_PROP_BACKLIGHT,
        cv2.CAP_PROP_ZOOM, cv2.CAP_PROP_FOCUS, cv2.CAP_PROP_PAN, cv2.CAP_PROP_TILT,
        cv2.CAP_PROP_IRIS
    ]
    for prop in properties:
        cap.set(prop, DEFAULT_VALUE)


def camera_reader(cap):
    """Lee frames de la cámara en un hilo separado."""
    global frame, ret, stop_thread
    while not stop_thread:
        try:
            ret, frame = cap.read()
            time.sleep(0.01)  # Pequeño retraso para reducir carga de CPU
        except Exception as e:
            print(f"Error al leer la cámara: {e}")
            stop_thread = True


def run_inference():
    """Carga el modelo YOLO, captura video y realiza inferencias."""
    global frame, ret, stop_thread

    print("Cargando modelo YOLO entrenado...")
    model = YOLO(MODEL_PATH)

    print("Iniciando captura de video...")
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Error: No se puede conectar a la cámara.")
        return

    configure_camera(cap)  # Configura la cámara con los parámetros

    # Inicia el hilo de lectura de la cámara
    camera_thread = threading.Thread(target=camera_reader, args=(cap,))
    camera_thread.start()

    print("Presiona 'q' para salir.")
    try:
        prev_time = time.time()  # Controla el tiempo entre inferencias
        while True:
            if not ret:
                continue  # Si no hay frame válido, sigue esperando

            # Limita la tasa de inferencia
            current_time = time.time()
            if current_time - prev_time >= INFERENCE_INTERVAL:
                results = model(frame)  # Realiza la inferencia
                prev_time = current_time

                # Anota y muestra el frame
                annotated_frame = results[0].plot()
                cv2.imshow('Detección en tiempo real', annotated_frame)

            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupción detectada. Cerrando...")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        # Cerrar correctamente la aplicación
        print("Finalizando...")
        stop_thread = True
        camera_thread.join()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
