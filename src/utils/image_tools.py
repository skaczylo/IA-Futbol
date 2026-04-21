"""
Modulo con funciones relativas al tratado de imagenes
"""
from turtle import color

from tqdm.notebook import tqdm
import cv2
import supervision as sv
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
import math

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

KEYPOINT_COLORS = []
for _ in range(57): # Generamos 57 colores
    # Mantenemos tu estilo de colores vibrantes / neón
    r = int(np.random.randint(200, 256))
    g = int(np.random.randint(0, 64))
    b = int(np.random.randint(0, 256))
    
    color_options = [(r, g, b), (r, b, g), (b, r, g), (b, g, r), (g, b, r), (g, r, b)]
    color_fijo = color_options[np.random.randint(0, 6)]
    
    KEYPOINT_COLORS.append(color_fijo)



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


def draw_annotations(image: np.ndarray, detections: sv.Detections, color: str = "#FF3300",stylized: bool = False) -> np.ndarray:
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
    if isinstance(color, str):
        color = sv.Color.from_hex(color)


    annotated_image = image.copy()
    annotator = sv.BoxAnnotator(color=color,thickness=2)
    
    if stylized:
        annotator = sv.EllipseAnnotator(color=color,thickness=2)
   
    # Aplicar el dibujo sobre la imagen
    annotated_image = annotator.annotate(scene=annotated_image,detections=detections)
    
    return annotated_image


def draw_keypoints(detections, image: np.ndarray, conf_threshold: float = 0.5) -> np.ndarray:
    """
    Dibuja los keypoints de las detecciones sobre la imagen de forma optimizada.
    Cada keypoint tiene un color fijo asignado.
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    confs = detections.keypoints.conf[0].cpu().numpy()
    kpts = detections.keypoints.xy[0].cpu().numpy()


    h, w = image.shape[:2]
    radius = max(1, int(w * 0.006))

    for i, (conf, kpt) in enumerate(zip(confs, kpts)):
        if conf < conf_threshold:
            continue

        x, y = int(kpt[0]), int(kpt[1])
        if x == 0 and y == 0:
            continue

        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)] 

        cv2.circle(image, (x, y), radius=radius, color=color, thickness=-1)

    return image




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
 


def annotate_yolo_pose(image_path: str, keypoints: dict) -> np.ndarray:
    """
    Dibuja las etiquetas YOLO Pose sobre una imagen.
 
    Args:
        image_path: Ruta a la imagen (.jpg, .png, ...)
        keypoints: Diccionario donde cada valor es (id, x_norm, y_norm, visibility)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en: {image_path}")

    h, w = img.shape[:2]


    print(keypoints)
    for x, y, v in keypoints.values():

        
        
        x_px = int(x * w)
        y_px = int(y * h)

        # Solo dibujar si el punto es visible (v == 2)
        if v == 2:
            

            # Color aleatorio para cada punto
            
            channels = [np.random.randint(200, 256), np.random.randint(0, 64), np.random.randint(0, 256)]
            np.random.shuffle(channels) # Esto modifica 'channels' directamente
            color = tuple(channels)     # Ahora 'channels' ya está mezclado

        
            cv2.circle(img, (x_px, y_px), radius=5, color=color, thickness=-1)

         
    return img





def _display_single_image_on_axis(ax, image_data):
    """
    Función auxiliar para leer (si es una ruta) y mostrar una imagen en un eje de matplotlib.
    Convierte de BGR a RGB para la correcta visualización.
    """
    img = None
    if isinstance(image_data, str):
        img = cv2.imread(image_data)
    elif isinstance(image_data, np.ndarray):
        img = image_data

    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
    
    ax.axis("off")


def display_image(image: np.ndarray | str, figsize=(15, 10), title: str = None):
    """
    Muestra una única imagen.
    
    Args:
        image (np.ndarray | str): Una única imagen (ruta o array de numpy).
        figsize (tuple): Tamaño total de la figura.
        title (str, opcional): Título de la imagen.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _display_single_image_on_axis(ax, image)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def display_images(images: list, cols: int, rows: int, titles: list = None, figsize: tuple = (15, 10)):
    """
    Muestra varias imágenes en una cuadrícula (grid) especificada.
    
    Args:
        images (list): Lista de imágenes (rutas o arrays).
        cols (int): Número de columnas en la cuadrícula.
        rows (int): Número de filas en la cuadrícula.
        titles (list, opcional): Lista de títulos para cada imagen.
        figsize (tuple): Tamaño común para cada imagen individual (ancho, alto).
    """
    num_images = len(images)
    if num_images == 0:
        print("La lista de imágenes está vacía.")
        return

    # Calcular el tamaño total de la figura basado en el tamaño asignado por imagen
    width_per_image, height_per_image = figsize
    total_width = cols * width_per_image
    total_height = rows * height_per_image

    fig, axes = plt.subplots(rows, cols, figsize=(total_width, total_height), squeeze=False)
    axes_flat = axes.flatten()

    for i, img_item in enumerate(images):
        if i >= len(axes_flat):
            break  # Evitar error si hay más imágenes que espacios en la cuadrícula
        
        _display_single_image_on_axis(axes_flat[i], img_item)
        
        if titles and i < len(titles):
            axes_flat[i].set_title(titles[i], fontsize=14)

    # Ocultar los ejes no utilizados si la cuadrícula no está llena por completo
    for j in range(num_images, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def fig_to_image(fig):
    """
    Convierte una figura de Matplotlib a una imagen de OpenCV (numpy array).
    
    Args:
        fig: Objeto de figura de Matplotlib.
        
    Returns:
        np.ndarray: Imagen en formato BGR (lista para OpenCV).
    """
    fig,ax = fig 
    fig.canvas.draw()

    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_bgr
