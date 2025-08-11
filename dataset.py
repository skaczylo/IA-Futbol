"""
Archivo .py con funciones que permiten la conversion o la transformacion de los datos a un formato con el que poder trabajar
Formato origininal:

- Deteccion/Tracking

    --Originial
        Archivo gt.txt. Formato de 10 caracteres:
            frame_id, track ID, (x,y) = esquina superior izquierda, ancho,alto, confidence score detection (siempre 1 en gt), -1,-1,-1,-1
    
    --Nuevo formato
        Formato de 5 caracteres
            class_id, x_center, y_center, ancho, altura

"""

import os
import shutil
import cv2 

def split_data(folder_path, new_path):
   
    folders = os.listdir(folder_path)

    for folder in folders:
        data = os.path.join(folder_path,folder,"gt","gt.txt")

        with open(data,"r") as gt:
            lines = gt.readlines()
            lines.sort(key=lambda l: int(l.split(',')[0])) #ordenar por primera columna
    
    
        num_image = 0
        write_folder= os.path.join(new_path,folder,"original_labels")
        os.makedirs(write_folder)

        i = 0
        while i < len(lines):

            num_image = num_image +1
            write_file = os.path.join(write_folder,f"{num_image:06d}.txt")

            with open(write_file,"w") as f:
                write = True
                while write:
                    f.write(lines[i])
                    i = i+1
                    if i ==len(lines) or num_image < int(lines[i].split(',')[0]):
                        write = False

def move_imgs(path,new_path):

    folders = os.listdir(path)
    for folder in folders:
        imgs = os.path.join(path,folder,"img1")
        new_imgs = os.path.join(new_path,folder)
        os.makedirs(new_imgs)
        shutil.move(imgs, new_imgs)



#Diccionario
def mapId(file):
    with open(file,"r") as f:
        line = f.readline()
        while(line[:5]!="track"):
            line = f.readline()
            pass
        
     
        map_id = {}
        num_object = 1

        while line :
            id = line.split('=')[1][1]
            if id =="p" or id =="g":
                map_id[num_object] = 1 #player
            elif id =="r":
                map_id[num_object] = 2 #referee
            else:
                map_id[num_object] = 0 #ball

            num_object = num_object +1
            line = f.readline()

    return map_id


def yolo_format_file(file,mapId,new_path,w_pixels,h_pixels):


    with open(file,"r") as f:
        original_file = f.readlines()
    
    with open(new_path,"w") as newf:
        for line in original_file:

            #Extraer datos
            line = line.split(',')
            
            class_id= int(line[1])
            x = int(line[2])
            y = int(line[3])
            width = int(line[4])
            height = int(line[5])

            #Clase
            newf.write(str(mapId[class_id]))
            newf.write(" ")

            #Calcular centro
            x_center = (x+ width/2)/w_pixels
            y_center = (y + height/2)/h_pixels

            newf.write(f"{x_center:.6f}")
            newf.write(" ")
            newf.write(f"{y_center:.6f}")
            newf.write(" ")
            
            #Ancho y alto
            newf.write(f"{width/w_pixels:.6f}")
            newf.write(" ")
            newf.write(f"{height/h_pixels:.6f}")
            newf.write("\n")



            


def yolo_format(path):

    folders = os.listdir(path)

    for folder in folders:

        #Extraer mapa de ids
        folder_path = os.path.join(path,folder)
        gameInfo_path = os.path.join(folder_path,"gameinfo.ini")
        ids = mapId(gameInfo_path)

        #Creamos nueva carpeta
        os.makedirs(os.path.join(folder_path,"labels"))
        new_folder = os.path.join(folder_path,"labels")

        original_labels = os.path.join(folder_path,"original_labels")
        labels = os.listdir(original_labels)

        for label in labels:

            label_name = label[0:6]+".jpg"
            image_asociated = os.path.join(folder_path,"img1",label_name)
            image_asociated = cv2.imread(image_asociated)

            height,width = image_asociated.shape[:2]

            new_label = os.path.join(new_folder,label)
            yolo_format_file(os.path.join(original_labels,label),ids,new_label,width,height)

    
   







