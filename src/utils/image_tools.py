"""
Modulo con funciones relativas al tratado de imagenes
"""
from tqdm.notebook import tqdm
import cv2
import supervision as sv
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt

# Ball = 0, Player = 1, Referee = 2
LABELS = ["Ball","Player","Referee"]
PLAYERS_AND_REFEREES = [1,2]
PLAYER = 1
REFEREE = 2
BALL = 0
BLUE = sv.Color(r=0, g=200, b =235)
RED = sv.Color(r=235, g=30, b=30)

TEAM_A = "#FF3300"
TEAM_B = "#0066FF"


def draw_annotations(
    image: np.ndarray, 
    detections: sv.Detections, 
    color: sv.Color = sv.Color.from_hex("#FF3300"), # ROJO por defecto
    stylized: bool = False
) -> np.ndarray:
    """
    Dibuja anotaciones sobre una imagen basadas en un objeto sv.Detections.
    
    Args:
        image (np.ndarray): La imagen o frame original.
        detections (sv.Detections): Objeto de supervision con las detecciones (bboxes, etc).
        color (sv.Color): Color a utilizar para las anotaciones.
        stylized (bool): Si es False, dibuja Bounding Boxes. Si es True, dibuja Elipses.
        
    Returns:
        np.ndarray: La imagen con las anotaciones dibujadas.
    """
   
    annotated_image = image.copy()
    annotator = sv.BoxAnnotator(color=color,thickness=2)
    
    if stylized:
        annotator = sv.EllipseAnnotator(color=color,thickness=2)
   
    # Aplicar el dibujo sobre la imagen
    annotated_image = annotator.annotate(scene=annotated_image,detections=detections)
    
    return annotated_image



def annotate_ball(image: np.array, detections: sv.Detections):
    """
    Dibuja un triangulo en la imagen sobre el balón detectado.
    
    Args:
        image (np.ndarray): La imagen o frame original.
        detections (sv.Detections): Objeto de supervision con las detecciones (bboxes, etc). 
    Returns:
        np.ndarray: La imagen con las anotaciones dibujadas.
    """
    annotated_image = image.copy()
    black_triangle = sv.TriangleAnnotator(color=sv.Color.BLACK,base=12,height=12) #marcado de la imagen
    annotated_image = black_triangle.annotate(scene=image.copy(),detections=detections)

    yellow_triangle = sv.TriangleAnnotator(color=sv.Color.YELLOW,base=10,height=10) #marcado de la imagen
    annotated_image = yellow_triangle.annotate(scene=annotated_image.copy(),detections=detections)

    return annotated_image


def crop_detections(image: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    """
    Recorta los objetos detectados en una imagen y los devuelve en una lista.
    
    Args:
        image (np.ndarray): La imagen original (frame) en formato BGR.
        detections (sv.Detections): Objeto de supervision con las detecciones.
        
    Returns:
        list[np.ndarray]: Lista de imágenes recortadas (crops). 
                          Si no hay detecciones, devuelve una lista vacía [].
    """

    crops = []
    
    for bbox in detections.xyxy:
       
        x_min, y_min, x_max, y_max = map(int, bbox)
    
        crop = image[y_min:y_max, x_min:x_max]  # Recortamos usando slicing de NumPy: image[y_inicio:y_fin, x_inicio:x_fin]
        
        crops.append(crop)
            
    return crops
        

def annotate_yolo(image_path: str , labels_path: str) -> np.ndarray:
    """
    Dibuja las etiquetas YOLO sobre una imagen.
 
    Args:
        image_path:  Ruta a la imagen (.jpg, .png, ...)
        labels_path: Ruta al .txt con etiquetas en formato YOLO
                     (class cx cy w h), coordenadas normalizadas [0, 1]
 
    Returns:
        Imagen anotada como array numpy BGR (misma que devuelve cv2)
    """
   
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
 
    # Paleta de colores por clase (BGR)
    rng = np.random.default_rng(42)
    colors = {i: tuple(int(c) for c in rng.integers(50, 220, 3)) for i in range(100)}
 
    with open(labels_path) as f:
        lines = [l.strip() for l in f if l.strip()]
 
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
 
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
 
        # Desnormalizar
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
 
        # Clamp para no salir de la imagen
        color = colors[class_id % 100]
 
        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
 
    return image
 



def display_image(image: np.array, figsize=(10, 8)):
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
