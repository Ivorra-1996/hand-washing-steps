# hand-washing-steps

## **Descripción del Proyecto**
En principio se basa en una IA que pueda detectar varios tipos de lavado de manos lo mejor posible...


## **Tecnologías Utilizadas**
- **YOLOv11**: Modelo de detección de objetos avanzado para reconocer manos y contar los dedos mostrados.
- **Python**: Lenguaje de programación principal para implementar la lógica del proyecto.
- **Visual Studio Code**: Entorno de desarrollo utilizado.
- **Cámara de celular**: Fuente de video en tiempo real.

## **Funcionamiento**
1. La cámara del celular transmite el video en tiempo real.
2. YOLOv11 procesa cada fotograma y detecta las manos visibles.
3. El modelo clasifica las manos según el tipo de lavado que haga (por ejemplo, Paso 1, Paso 2 o Paso 3, etc).
4. El sistema muestra en pantalla el valor correspondiente a la clase detectada.

## **Instalación**
1. Clona el repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd hands-counter

### Problema: Persistencia de rutas antiguas en Ultralytics YOLO

#### Descripción
Al ejecutar el entrenamiento con Ultralytics YOLO, puede ocurrir un error relacionado con una ruta antigua persistente en la configuración global de Ultralytics. El error se ve algo similar a esto:

FileNotFoundError: Dataset 'data.yaml' images not found ⚠️, missing path 'C:\Users\Usuario\proyecto-antiguo\datasets\valid\images'

Aunque el proyecto actual está en una ruta diferente, Ultralytics sigue intentando usar configuraciones antiguas guardadas en `settings.yaml`.

#### Solución
1. **Ubica el archivo global `settings.yaml`**:  
   Se encuentra en la siguiente ruta:
        C:\Users<Usuario>\AppData\Roaming\Ultralytics\settings.yaml

2. **Edita el archivo y actualiza la ruta**:  
Busca la línea que contiene `dataset_dir` y actualízala con la ruta del proyecto actual:
```yaml
dataset_dir: C:\Users\<Usuario>\Desktop\hands-counter\hands-counter
```

3. **Otra opción**:
    Borrar el archivo `settings.yaml`, despues se crea otro solo cuando ejecutas el entrenamiento.


