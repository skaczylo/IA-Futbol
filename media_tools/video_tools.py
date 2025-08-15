from tqdm import tqdm
import cv2
import supervision as sv
import numpy as np
from IPython.display import HTML
from . import image_tools 

# Ball = 0, Player = 1, Referee = 2
LABELS = ["Ball","Player","Referee"]
PLAYERS_AND_REFEREES = [1,2]
PLAYER = 1
REFEREE = 2
BALL = 0
BLUE = sv.Color(r=0, g=200, b =235)

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
        frame_ = image_tools.annotate_frame(frame,
                                            detections=sv.Detections.from_ultralytics(results[0]),
                                            stylized=stylized_ob_detection,
                                            text_label=text_label)
        video_BB.write(frame_)
          
        
    
    video.release()
    video_BB.release()