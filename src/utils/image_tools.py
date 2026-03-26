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
    # Hacemos una copia para no modificar la imagen original por referencia
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
        

def yolo_to_supervision_bb(img_size,bb_yolo):
    """
    Transforma las coordenadas del BB en formato YOLO (x_centro, y_centro,ancho,altura) normalizaod
    a un formato válido para supervision : xyxy = esquina superior izquierda + esquina inferior derecha

    """
    img_height,img_width, _ = img_size
    x_c, y_c, w, h = bb_yolo
    x_center = x_c * img_width
    y_center = y_c * img_height
    box_width = w * img_width
    box_height = h * img_height

    x_1 = int(x_center - box_width / 2)
    y_1 = int(y_center - box_height / 2)
    x_2 = int(x_center + box_width / 2)
    y_2 = int(y_center + box_height / 2)

    return (x_1, y_1, x_2, y_2)



def display_image(image: np.array, figsize=(10, 8)):
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
