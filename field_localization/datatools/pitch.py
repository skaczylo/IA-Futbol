import matplotlib.pyplot as plt
import numpy as np
from baseline.soccerpitch import SoccerPitch
import datatools.ellipse
import cv2
from typing import List, Tuple
from  datatools.ellipse import get_pitch
import numpy as np
from datatools.ellipse import INTERSECTON_TO_PITCH_POINTS
import baseline.soccerpitch  as soccerpitch

background_color = (34, 139, 34)  # Verde bosque en BGR
line_color = (255, 255, 255)      # Blanco en BGR
padding: int = 5  # Margen mínimo
line_thickness: int = 3
point_radius: int = 4
scale = 4

def m_to_px(point):
        px_x = int((point[0] + soccerpitch.PITCH_LENGTH / 2) * scale) 
        px_y = int((point[1] + soccerpitch.PITCH_WIDTH / 2) * scale) 
        return (px_x, px_y)


def pitch_picture():
    pitch_info = SoccerPitch()
    
    # 1. Dimensiones ajustadas por la escala
    scaled_length = int(pitch_info.PITCH_LENGTH * scale)
    scaled_width = int(pitch_info.PITCH_WIDTH * scale)

    # 2. Creamos el lienzo y lo rellenamos con el color de fondo
    pitch_image = np.ones(
        (scaled_width, scaled_length, 3),
        dtype=np.uint8
    ) * np.array(background_color, dtype=np.uint8)

    
    # 4. Dibujar líneas y arcos
    field_elements = pitch_info.sample_field_points()
    
    for element_name, points in field_elements.items():
        # Saltamos los postes de la portería
        if "post" in element_name.lower() or "crossbar" in element_name.lower():
            continue
            
        for i in range(len(points) - 1):
            pt1 = m_to_px(points[i])
            pt2 = m_to_px(points[i+1])
            cv2.line(
                pitch_image, pt1, pt2, 
                color=line_color,          # <-- Usamos la tupla BGR directamente
                thickness=line_thickness,
                lineType=cv2.LINE_AA
            )

    # 5. Dibujar puntos clave (Penaltis y Centro)
    for spot_key in ["L_PENALTY_MARK", "R_PENALTY_MARK", "CENTER_MARK"]:
        if spot_key in pitch_info.point_dict:
            cv2.circle(
                pitch_image,
                center=m_to_px(pitch_info.point_dict[spot_key]),
                radius=point_radius,
                color=line_color,          # <-- Usamos la tupla BGR directamente
                thickness=-1,
                lineType=cv2.LINE_AA
            )

    return pitch_image


PITCH_PICTURE= pitch_picture()


#=================================
#Funciones Homografía
#=================================


def get_homography(puntos_detectados):
    """
    Calcula la matriz de homografía 3x3 que transforma píxeles de la imagen a metros del campo.
    """
    src_pts = [] # Píxeles de la imagen
    dst_pts = [] # Metros reales

    PITCH_POINTS = get_pitch()
    
    # Recorremos los puntos detectados
    for point_id, coords_img in puntos_detectados.items():
        if coords_img is not None:
            
            # 2. Buscamos el equivalente en el campo real
            if point_id in INTERSECTON_TO_PITCH_POINTS:
                point_name = INTERSECTON_TO_PITCH_POINTS[point_id]
                
                if point_name in PITCH_POINTS:
                    coords_real = PITCH_POINTS[point_name]
                    
                    # 3. Guardamos los pares alineados
                    src_pts.append([coords_img[0], coords_img[1]])
                    dst_pts.append([coords_real[0], coords_real[1]]) # Ignoramos la Z
    
    
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    
    if len(src_pts) < 4:
        return None
    
    # 6. Calculamos la matriz usando RANSAC (filtra "outliers" o puntos mal detectados)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H

def proyectar_jugadores(boxes, H):
    """
    Toma las cajas de YOLO (results.boxes) y la matriz H.
    Devuelve una lista de coordenadas (x, y) en METROS del campo real.
    """
    jugadores_metros = []
    
    if H is None or boxes is None:
        return jugadores_metros
        
    for box in boxes:
        # Extraer coordenadas de la caja [x_min, y_min, x_max, y_max]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        
        # Calcular los pies del jugador (Centro en X, Base en Y)
        x_pies = (x1 + x2) / 2.0
        y_pies = y2 
        
        # Formato estricto que pide OpenCV para transformar puntos
        punto_pixel = np.array([[[x_pies, y_pies]]], dtype=np.float32)
        
        # ¡La magia de la homografía! Transformamos a metros
        punto_metro = cv2.perspectiveTransform(punto_pixel, H)
        
        # Extraer la X y la Y reales
        x_m, y_m = punto_metro[0][0]
        jugadores_metros.append((x_m, y_m))
        
    return jugadores_metros


def draw_2Dpitch(points_detected=None, players_points=None):
    
    if points_detected is None or players_points is None:
        return pitch_image
    

    pitch_image = PITCH_PICTURE.copy()


    H = get_homography(points_detected)

    if H is None:
        return pitch_image
    
    # IMPORTANTE: Pasamos H a la función de proyectar
    jugadores_metros = proyectar_jugadores(players_points, H)

    
    if jugadores_metros is not None:
        for jug_coords in jugadores_metros:
            # Los jugadores ya vienen en metros gracias a tu función proyectar_jugadores
            px_coords = m_to_px(jug_coords)
            
            # Dibujar círculo azul un poco más grande para los jugadores
            cv2.circle(pitch_image, px_coords, radius=5, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            
            # Opcional: Borde negro alrededor del jugador para que resalte más
            cv2.circle(pitch_image, px_coords, radius=5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return pitch_image






















