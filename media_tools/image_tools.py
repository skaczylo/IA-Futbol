"""
Modulo con funciones relativas al tratado de imagenes

"""

from tqdm import tqdm
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



def referee_annotator(image,detections,text_label =True):
    """
    Marca a los arbitros con una elipse 
    """

    ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.WHITE) #marcado de la imagen
    annotated_image = ellipse_annotator.annotate(scene=image.copy(),detections=detections)

    if text_label is True:
        label_annotator = sv.LabelAnnotator(color=sv.Color.WHITE,text_color=sv.Color.BLACK,text_padding=3,text_position=sv.Position.BOTTOM_CENTER) #etiqueta de clase
        labels = ["Referee"] * len(detections.class_id) #Tantas etiquetas como arbitros
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    return annotated_image

def player_annotator(image,detections,text_label =True):
    """
    Marca a los jugadores con una elipse
    """

    ellipse_annotator = sv.EllipseAnnotator(color=BLUE) #marcado de la imagen
    annotated_image = ellipse_annotator.annotate(scene=image.copy(),detections=detections)

    if text_label is True:
    
        label_annotator = sv.LabelAnnotator(color=BLUE,text_color=sv.Color.WHITE,text_padding=3,text_position=sv.Position.BOTTOM_CENTER) #etiqueta de clase
        labels = ["Player"] * len(detections.class_id) #Tantas etiquetas como jugadores
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
   
    return annotated_image


def ball_annotator(image,detections):
    """
    Marca el balon de futbol con un triangulo amarillo
    """
    #Dos triangulos, uno negro y otro amarillo encima mas pequeño para hacer un marcado de la base del triangulo y que sea mas visible

    black_triangle = sv.TriangleAnnotator(color=sv.Color.BLACK,base=12,height=12) #marcado de la imagen
    annotated_image = black_triangle.annotate(scene=image.copy(),detections=detections)

    yellow_triangle = sv.TriangleAnnotator(color=sv.Color.YELLOW,base=10,height=10) #marcado de la imagen
    annotated_image = yellow_triangle.annotate(scene=annotated_image.copy(),detections=detections)

    return annotated_image






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

def annotate_frame(image,detections,stylized=False,text_label = True):
    """
    Dada una imagen ya cargada y una deteccion en formato Supervision, añade los correspondientes BB sobre la imagen
    No modifica la imagen original, devuelve una nueva imagen con las etiquetas
    """

    if stylized == True:
        ball_detection = detections[detections.class_id == BALL]
        player__detections = detections[detections.class_id == PLAYER]
        referee_detections = detections[detections.class_id == REFEREE]

        frame_ = ball_annotator(image,ball_detection)
        frame_ = player_annotator(frame_,player__detections,text_label)
        frame_ = referee_annotator(frame_,referee_detections,text_label)
    else:
        bounding_box_annotator = sv.BoxAnnotator()
        frame_ = bounding_box_annotator.annotate(scene=image.copy(),detections=detections)

    return frame_


def annotate_image(image,labels,stylized = False):
    """
    Dada una ruta de imagen, la función dibuja las etiquetas en formato yolo (labels) sobre la imagen.
    No modifica la imagen original, devuelve una nueva imagen con las correspondientes etiquetas.
    """

    #Archivos
    img = cv2.imread(image) #Abrir imagen
    with open(labels,"r") as f:
        labels = f.readlines()

    #Datos
    xyxy = []
    class_ids = []
    
    #Extraer y adaptar etiquetas
    for label in labels:
        data = label.split() #Formato YOLO : class_id x_center y_center w h
        class_id, *bb_yolo = data

        class_id = int(class_id)
        bb_yolo = [float(i) for i in bb_yolo]
        bb_sv = yolo_to_supervision_bb(img.shape,bb_yolo)
        xyxy.append(bb_sv)
        class_ids.append(class_id)

    #Anotacion de imagen
    labels_sv = sv.Detections(np.array(xyxy),class_id=np.array(class_ids)) #Objeto que permite trabajar con distintos tipos de anotaciones
    annotated_image = annotate_frame(img,labels_sv,stylized=stylized)

    return annotated_image

def display_image(image, figsize=(10, 8)):
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()










    