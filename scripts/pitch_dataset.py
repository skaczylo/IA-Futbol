"""
Modifica el dataset original de Soccernet generando las etiquetas y las coordenadas de los keypoints.

El nuevo formato se ajustará para entrenar un modelo YOLOPose de puntos claves o keypoints.
"""

import cv2
import os
import shutil
from pitch_keypoints.intersections import get_intersections #Puntos del campo
from pitch_keypoints.reader import read_annot
from  tqdm.auto import tqdm
from utils.config import PITCH_KEYPOINTS_DATA


# Se asume la disponibilidad de: read_annot, get_intersections

def generate_soccernet_yolo_pose(source_path: str, output_path: str):
    """
    Crea un dataset YOLO Pose procesando archivos .json y .jpg desde una misma carpeta.
    Calcula las intersecciones a partir de las anotaciones JSON y genera archivos .txt con formato YOLO Pose para cada imagen, además de copiar las imágenes a la carpeta de destino.
    El formato del archivo .txt es:
        - Una fila por objeto. En nuestro caso solo hay un objeto: el campo que será la imagen entera.
        - En cada fila:
            <class-index> <x> <y> <width> <height> <px1> <py1> <vis1><px2> <py2> ... <pxn> <pyn> <visn>

    Args:
        source_path: Carpeta que contiene los archivos .jpg y .json originales.
        output_path: Carpeta donde se creará la estructura 'images' y 'labels'.
    """
    #Carpetas subset/Images y subset/labels
    images_dst = os.path.join(output_path, "images")
    labels_dst = os.path.join(output_path, "labels")
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    
    json_files = [f for f in os.listdir(source_path) if f.endswith('.json')]


    for json_name in tqdm(json_files, desc="Procesando archivos"):

        file_id = os.path.splitext(json_name)[0] #nombre del archivo sin extensión

        #Soccernet files
        json_src_path = os.path.join(source_path, json_name)
        image_src_path = os.path.join(source_path, f"{file_id}.jpg")

    
        img = cv2.imread(image_src_path)
        h, w = img.shape[:2]


        #Calcular intersecciones a partir de anotaciones JSON
        annots = read_annot(json_src_path) #
        intersections, _ = get_intersections(annots, img_size=(w, h)) #

        
        yolo_line = ["0", "0.5", "0.5", "1.0", "1.0"] #El campo será toda la imagen, por lo que la caja delimitadora es el tamaño completo de la image

        # El orden de los keypoints debe ser estrictamente del 0 al 56
        for i in range(57):
            point = intersections.get(i)
            
            if point is not None:
                x_norm = point[0] / w
                y_norm = point[1] / h
                yolo_line.extend([f"{x_norm:.6f}", f"{y_norm:.6f}", "2"])
            else:
                # Punto no visible o no calculado: se marca con visibilidad 0
                yolo_line.extend(["0.000000", "0.000000", "0"])

        
        with open(os.path.join(labels_dst, f"{file_id}.txt"), "w") as f:
            f.write(" ".join(yolo_line) + "\n")

        
        shutil.copy(image_src_path, os.path.join(images_dst, f"{file_id}.jpg"))


if __name__ == "__main__":

    train_path = os.path.join(PITCH_KEYPOINTS_DATA, "train")
    test_path = os.path.join(PITCH_KEYPOINTS_DATA, "test")
    val_path = os.path.join(PITCH_KEYPOINTS_DATA, "val")

    #Creacion carpetas
    os.makedirs(train_path, exist_ok=True) #Creamos nueva carpeta
    os.makedirs(test_path, exist_ok=True) #Creamos nueva carpeta
    os.makedirs(val_path, exist_ok=True) #Creamos nueva carpeta

    #Keypoints

    #generate_soccernet_yolo_pose(os.path.join(PITCH_KEYPOINTS_DATA,"soccernet","train"), train_path)
    #generate_soccernet_yolo_pose(os.path.join(PITCH_KEYPOINTS_DATA,"soccernet","test"), test_path)
    generate_soccernet_yolo_pose(os.path.join(PITCH_KEYPOINTS_DATA,"soccernet","valid"), val_path)




    
