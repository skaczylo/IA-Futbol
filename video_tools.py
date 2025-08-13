from tqdm import tqdm
import cv2
import supervision as sv
import numpy as np

# Ball = 0, Player = 1, Referee = 2
PLAYERS_AND_REFEREES = [1,2]
BALL = 0

def bb_annotator(image,detections):
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    annotated_frame = bounding_box_annotator.annotate(scene=image.copy(),detections=detections)

    return annotated_frame

def person_annotator(image,detections):
    triangle_annotator = sv.EllipseAnnotator()
    annotated_frame = triangle_annotator.annotate(scene=image.copy(),detections=detections)
    return annotated_frame


def ball_annotator(image,detections):
    triangle_annotator = sv.TriangleAnnotator()
    annotated_frame = triangle_annotator.annotate(scene=image.copy(),detections=detections)
    return annotated_frame



def write_video(video_path,output_path,object_model,stylized_ob_detection):
    
    #Extraemos propiedades
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_BB =  cv2.VideoWriter(output_path,fourcc,fps,(width,height),isColor=True)
  

    for _ in tqdm(range(total_frames),desc="Processing"):

        #Lee el frame
        ret, frame = video.read()
        
        if not ret:
            break
        
        #Marcado de Objetos
        results = object_model(frame,verbose=False) #devuelve una lista: en este caso una lista con un solo objeto
        detections  = sv.Detections.from_ultralytics(results[0])

        if stylized_ob_detection == True:
            ball_detection = detections[detections.class_id == BALL]
            player_referee_detection = detections[np.isin(detections.class_id,PLAYERS_AND_REFEREES)]

            frame_ = ball_annotator(frame,ball_detection)
            frame_ = person_annotator(frame_,player_referee_detection)
        else:
            frame_ = bb_annotator(frame,detections)


        video_BB.write(frame_)
          
        
    
    video.release()
    video_BB.release()