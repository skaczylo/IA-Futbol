"""
Modifica el dataset original de Soccernet generando las etiquetas y las coordenadas de los keypoints
"""

import cv2
import os
import shutil
from datatools.intersections import get_intersections #Puntos del campo
from datatools.reader import read_annot
import json
from  tqdm import tqdm

def organize_dataset(folder_path):
    """
    Organiza un dataset de fútbol moviendo imágenes a 'images' y JSON a 'soccernet_labels'.
    
    Args:
        folder_path (str): Ruta a la carpeta del dataset (por ejemplo, 'dataset/train')
    """
    # Rutas destino
    images_dir = os.path.join(folder_path, 'images')
    labels_dir = os.path.join(folder_path, 'soccernet_labels')
    
    # Crear carpetas si no existen
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Iterar archivos en la carpeta principal
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ignorar carpetas
        if os.path.isdir(file_path):
            continue
        
        # Mover imágenes
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.move(file_path, os.path.join(images_dir, filename))
        
        # Mover JSON
        elif filename.lower().endswith('.json'):
            shutil.move(file_path, os.path.join(labels_dir, filename))
            
    print(f"Organización completada en '{folder_path}'.")
    print(f"Imágenes → {images_dir}")
    print(f"JSON → {labels_dir}")


def new_yolo_labels(folder_path):

    box_size = 15.0  #Pixeles

    yolo_folder = os.path.join(folder_path,"labels")
    img_folder = os.path.join(folder_path,"images")
    os.makedirs(yolo_folder, exist_ok=True) #Creamos nueva carpeta


    for json in tqdm(os.listdir(os.path.join(folder_path,"soccernet_labels")),"Processing"): #Leemos jsons


        id = os.path.splitext(json)[0]
        json_path = os.path.join(folder_path,"soccernet_labels",json) #Ruta completa al json n-esimo
        image_path = os.path.join(img_folder,id+".jpg") #Ruta completa a la imagen n-esima



        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        annots = read_annot(json_path)
        intersections, _ = get_intersections(annots)

        line = [] #etiqueta : Clase x y widht height px1 py1 vis1 px2 py2 vis2

        line.append("0")
        line.append(f"{0.5}")
        line.append(f"{0.5}")
        line.append(f"{1}")
        line.append(f"{1}")
        
        for class_id, pt in intersections.items():
            if pt is not None:
               
                x_px, y_px = float(pt[0]), float(pt[1])
                
                # Normalizar el centro de la caja (dividiendo por ancho y alto de la imagen)
                x_center_norm = x_px / img_w
                y_center_norm = y_px / img_h
                
          
                # Formato YOLO Object Detection: <class> <x> <y> <w> <h>
                line.append(f"{x_center_norm:.6f} {y_center_norm:.6f} {1}")
            
            else:
                line.append(f"{0:.6f} {0:.6f} {0}")



        # 4. Escribir el archivo .txt solo si hay puntos visibles
        
        new_yolo_label = os.path.join(yolo_folder, id + '.txt')
        with open(new_yolo_label, 'w') as txt_file:
            txt_file.write(" ".join(line) + "\n")


def new_labels(folder_path):
   

    new_folder = os.path.join(folder_path,"new_labels")
    img_folder = os.path.join(folder_path,"images")
    os.makedirs(new_folder, exist_ok=True) #Creamos nueva carpeta


    for json_f in os.listdir(os.path.join(folder_path,"soccernet_labels")): #Leemos jsons


        id = os.path.splitext(json_f)[0]
        json_path = os.path.join(folder_path,"soccernet_labels",json_f) #Ruta completa al json n-esimo
        image_path = os.path.join(img_folder,id+".jpg") #Ruta completa a la imagen n-esima



        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        def new_label(item):
            clase, keypoint = item # Desempaquetamos la clave y los valores

            if keypoint is not None:
                x,y = keypoint
                x_norm = round(x , 4) #4 decimales
                y_norm = round(y , 4)
                return clase, {"Keypoint":(x_norm,y_norm) ,"Visibility": 1} #Clase, coordendas, visible
            else:
                return clase, {"Keypoint":(0,0) ,"Visibility": 0}


        annots = read_annot(json_path)
        intersections, _ = get_intersections(annots)

        keypoints = dict(map(new_label,intersections.items()))

        with open(os.path.join(new_folder, f'{id}.json'), "w") as f:
            json.dump(keypoints,f,indent = 2)

        
new_yolo_labels("pitch_keypoints/dataset/train")
new_yolo_labels("pitch_keypoints/dataset/valid")
new_yolo_labels("pitch_keypoints/dataset/test")




    
