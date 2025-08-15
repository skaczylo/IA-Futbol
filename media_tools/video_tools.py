from tqdm import tqdm
import cv2
import supervision as sv
import numpy as np
from IPython.display import HTML
from . import image_tools as it
import random
from tqdm import tqdm

# Ball = 0, Player = 1, Referee = 2
LABELS = ["Ball","Player","Referee"]
PLAYERS_AND_REFEREES = [1,2]
PLAYER = 1
REFEREE = 2
BALL = 0
BLUE = sv.Color(r=0, g=200, b =235)



def extract_crops(video_path,object_model,ids,crop_number=750):
    """
    Dado un video, extraerá, en total, "crop_numer" recortes de las clases indicadas en "class_ids"
    Para ello seleccionará frames aleatorios del video hasta conseguir la cantidad deseada
    """

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    crops = []

    progres_bar = tqdm(total=crop_number) #Barra de progreso

    while len(crops) < crop_number:

        random_frame =random.randint(0,total_frames-1)
        video.set(cv2.CAP_PROP_POS_FRAMES, random_frame)

        ret,frame = video.read()

        if ret: #Lectura correcta 
            results = object_model(frame,verbose=False)

            detections = sv.Detections.from_ultralytics(results[0]) #Facilita mucho el filtrado de la clase
            detections = detections[np.isin(detections.class_id, ids)]

            new_crops = it.extract_crops(frame,detections.xyxy)

            crops.extend(new_crops)
            progres_bar.update(len(new_crops))

    progres_bar.close()

    return crops



def display_video(video_path,height=450,width=800):
    """
    Muestra el video en el notebook mediante HTML
    """


    return HTML(f"""
    <video width="{width}" height="{height}" controls>
        <source src="{video_path}" type="video/mp4">
    </video>
    """)

def write_video(video_path,output_path,object_model,stylized_ob_detection,text_label = True):
    
    #Extraemos propiedades
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_BB =  cv2.VideoWriter(output_path,fourcc,fps,(width,height),isColor=True)
  
    for _ in tqdm(range(total_frames),desc="Processing"):

        #Lee el frame
        ret, frame = video.read()
        
        if not ret:
            break
        
        #Marcado de Objetos
        results = object_model(frame,verbose=False) #devuelve una lista, en este caso una lista con un solo objeto
        frame_ = it.annotate_image(frame,
                                    labels=sv.Detections.from_ultralytics(results[0]),
                                    stylized=stylized_ob_detection,
                                    text_label=text_label)
        video_BB.write(frame_)
          
        
    
    video.release()
    video_BB.release()