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


def bounding_box_annotator(image,detections,text_label = True):

    bounding_box_annotator = sv.BoxAnnotator()
    frame_ = bounding_box_annotator.annotate(scene=image.copy(),detections=detections)
    
    if text_label is True:
        
        label_annotator = sv.LabelAnnotator()
        labels = [LABELS[int(id)] for id in detections.class_id]
        frame_ = label_annotator.annotate(scene=frame_, detections=detections, labels=labels)

    return frame_

def ellipse_annotator(image,detections,text_label = True):
    ellipse_annotator = sv.EllipseAnnotator() #marcado de la imagen
    annotated_image = ellipse_annotator.annotate(scene=image.copy(),detections=detections)

    labels = [LABELS[int(id)] for id in detections.class_id]

    if text_label is True:
        label_annotator = sv.LabelAnnotator(text_padding=3,text_position=sv.Position.BOTTOM_CENTER) #etiqueta de clase
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    return annotated_image

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

def player_annotator(image,detections,text_label =True, team = 0):
    """
    Marca a los jugadores con una elipse
    """
    if team == 0:
        ellipse_annotator = sv.EllipseAnnotator(color=RED) #marcado de la imagen
        annotated_image = ellipse_annotator.annotate(scene=image.copy(),detections=detections)
    else:
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



def extract_crops(image, xyxy):
    """
    Dada una imagen y una lista de recortes, devuelve una lista de imagenes en ndarray correspondientes 
    a los recortes
    """

    if not isinstance(image, np.ndarray):
        image = cv2.imread()

    crops = []
    for coordinates in xyxy:
        x1,y1,x2,y2 = coordinates

        crop = image[int(y1):int(y2), int(x1):int(x2)]
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


def annotate_image(image: str | np.ndarray,labels: str,stylized=False, text_label = True):
    """
    Anota las imagenes de una imagen
    "image" puede ser tanto una ruta a una imagen  o una imagen en ndarray
    Además "labels" ha de ser una ruta a un .txt con etiquetas en formato YOLO o
    un objeto Detections de la libreria Supervision
    """

    labels_sv = labels
    #Adaptacion de los datos
    if not isinstance(image, np.ndarray): #Es una ruta a una imagen
        image = cv2.imread(image)

    if isinstance(labels,str): #Es un fichero en formato YOLO
        #Necesitamos leer el fichero y convertir los datos en un formato valido para Supervision (sv)

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
            bb_sv = yolo_to_supervision_bb(image.shape,bb_yolo)
            xyxy.append(bb_sv)
            class_ids.append(class_id)

        labels_sv = sv.Detections(np.array(xyxy),class_id=np.array(class_ids))
    

    #Marcado de la imagen
    if stylized == True:
        ball_detection = labels_sv[labels_sv.class_id == BALL]
        player__detections = labels_sv[labels_sv.class_id == PLAYER]
        referee_detections = labels_sv[labels_sv.class_id == REFEREE]

        frame_ = ball_annotator(image,ball_detection)
        frame_ = player_annotator(frame_,player__detections,text_label)
        frame_ = referee_annotator(frame_,referee_detections,text_label)
    else:

        frame_ = bounding_box_annotator(image,labels_sv,text_label=text_label)
        

    return frame_


def display_image(image, figsize=(10, 8)):
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()










    