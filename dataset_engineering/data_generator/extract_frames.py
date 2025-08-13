"""
Extrae frames de los videos descargados y los almacena en el dataset para su posterior etiquetado

"""
import os
import cv2 as cv
import numpy as np
import time
import uuid

RESULT_PATH ="dataset_engineering/data_generator/generated_images"
VIDEOS_PATH = "dataset_engineering/data_generator/videos"


def extract_frames(video_path,output_path,percentage = 0.5,delete=True):
    """
    Extrae del video dado un 20% de los frames y los almacena en la ruta indicada por output_path.
    Ademas, por defecto, elimina el video una vez extraida las imagenes
    """

    video = cv.VideoCapture(video_path)
    fps = video.get(cv.CAP_PROP_FPS) #Obtener fps
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    total_seconds = total_frames/fps
    number_seconds = int(percentage *(total_seconds-20)) #numero de frames que queremos obtener: un 20% por defecto

    selected_seconds = np.linspace(10, total_seconds-10, number_seconds,dtype=int)

    for second in selected_seconds:
        video.set(cv.CAP_PROP_POS_FRAMES, second*fps)
        ret, frame = video.read()

        if not ret or frame is None:
            print(f"Failed to read frame at {second}s in {video_path}")
            continue

        frame_id = f"{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:10] }" # formato tipo 20250710_155203;parte aleatoria para evitar duplicados
        cv.imwrite(os.path.join(output_path,f"{frame_id}.jpg"),frame)


        
    video.release()
    if delete:
        os.remove(video_path)





    






