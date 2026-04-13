import numpy as np
from ia_futbol.pitch_keypoints.ellipse import INTERSECTON_TO_PITCH_POINTS
from ia_futbol.pitch_keypoints.ellipse import PITCH_POINTS
import cv2


def get_homography(keypoints_detected,conf = 0.85):

    """
    Calcula, si se puede, Homografia H de pixeles a mapa 2D
    """

    H = None
    if len(keypoints_detected) > 1:
        keypoints_detected = keypoints_detected[0]

    keypoints_detected = keypoints_detected.squeeze(0)

    src_points = [] #Puntos de la imagen
    trg_points = [] #Puntos del campo 2D

    for i, kpt in enumerate(keypoints_detected):

        x,y,kpt_conf = kpt

        if kpt_conf >= conf:


            kpt_cls = INTERSECTON_TO_PITCH_POINTS[i]
            src_points.append([int(x),int(y)])
            trg_points.append(PITCH_POINTS[kpt_cls])


    #Formato especifico cv2
    src_points = np.array(src_points, dtype=np.float32)
    trg_points = np.array(trg_points, dtype=np.float32)

    if len(src_points) >= 4:
        H, status = cv2.findHomography(src_points, trg_points, cv2.RANSAC, 5.0)
        
    return H