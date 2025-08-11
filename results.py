import cv2
import time


from ultralytics import YOLO

# Carga el modelo entrenado desde el archivo .pt
model = YOLO('runs/detect/train/weights/best.pt')

# load video
video_path = 'Dataset/videos_prueba/out.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model(frame)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break