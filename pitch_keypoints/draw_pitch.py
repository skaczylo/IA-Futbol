import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import numpy as np
import cv2
import torch

# Asumo que tienes definido GREEN en tu entorno, por ejemplo: GREEN = '#4C9A2A'
GREEN = '#4C9A2A'
def draw_pitch(figsize = (8.5,4.8)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color de fondo verde estilo césped
    fig.patch.set_facecolor(GREEN)
    ax.set_facecolor(GREEN)

    # --- LÍNEAS PRINCIPALES ---
    # Borde exterior del campo (de -52.5 a 52.5 en X, y de -34 a 34 en Y)
    ax.plot([-52.5, -52.5, 52.5, 52.5, -52.5], [-34, 34, 34, -34, -34], color="white", linewidth=2, zorder=2)
    # Línea de medio campo
    ax.plot([0, 0], [-34, 34], color="white", linewidth=2, zorder=2)

    # --- CÍRCULO CENTRAL Y PUNTO ---
    circulo_central = Circle((0, 0), 9.15, color="white", fill=False, linewidth=2, zorder=2)
    ax.add_patch(circulo_central)
    ax.plot(0, 0, "o", color="white", markersize=4, zorder=2) # Punto central ahora es 0,0

    # --- ÁREAS GRANDES (PENALTI) ---
    # Izquierda
    ax.plot([-52.5, -36.0, -36.0, -52.5], [-20.16, -20.16, 20.16, 20.16], color="white", linewidth=2, zorder=2)
    # Derecha
    ax.plot([52.5, 36.0, 36.0, 52.5], [-20.16, -20.16, 20.16, 20.16], color="white", linewidth=2, zorder=2)

    # --- ÁREAS PEQUEÑAS (META) ---
    # Izquierda
    ax.plot([-52.5, -47.0, -47.0, -52.5], [-9.16, -9.16, 9.16, 9.16], color="white", linewidth=2, zorder=2)
    # Derecha
    ax.plot([52.5, 47.0, 47.0, 52.5], [-9.16, -9.16, 9.16, 9.16], color="white", linewidth=2, zorder=2)

    # --- PUNTOS DE PENALTI ---
    ax.plot(-41.5, 0, "o", color="white", markersize=3, zorder=2)  # Izquierda
    ax.plot(41.5, 0, "o", color="white", markersize=3, zorder=2)   # Derecha

    # --- ARCOS DE PENALTI (Semicírculos fuera del área) ---
    # Izquierda
    arco_izq = Arc((-41.5, 0), 18.3, 18.3, angle=0, theta1=-53, theta2=53, color="white", linewidth=2, zorder=2)
    # Derecha
    arco_der = Arc((41.5, 0), 18.3, 18.3, angle=0, theta1=127, theta2=233, color="white", linewidth=2, zorder=2)
    ax.add_patch(arco_izq)
    ax.add_patch(arco_der)

    # --- ARCOS DE CÓRNER ---
    ax.add_patch(Arc((-52.5, -34), 2, 2, angle=0, theta1=0, theta2=90, color="white", linewidth=2, zorder=2))
    ax.add_patch(Arc((-52.5, 34), 2, 2, angle=0, theta1=270, theta2=360, color="white", linewidth=2, zorder=2))
    ax.add_patch(Arc((52.5, -34), 2, 2, angle=0, theta1=90, theta2=180, color="white", linewidth=2, zorder=2))
    ax.add_patch(Arc((52.5, 34), 2, 2, angle=0, theta1=180, theta2=270, color="white", linewidth=2, zorder=2))

    # --- PORTERÍAS ---
    ax.plot([-52.5, -54.5, -54.5, -52.5], [-3.66, -3.66, 3.66, 3.66], color="white", linewidth=2, zorder=2)
    ax.plot([52.5, 54.5, 54.5, 52.5], [-3.66, -3.66, 3.66, 3.66], color="white", linewidth=2)
    
    # Mantener las proporciones y quitar ejes
    ax.set_aspect('equal') 
    ax.axis('off')
    ax.set_xlim(-54.5, 54.5)
    ax.set_ylim(-35, 35)
   
    ax.invert_yaxis()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    #fig.canvas.draw()


    return fig, ax

def get_pitch_image(players_keypoints = None, team_segmentation = None):

    fig , ax  = draw_pitch()
    
    if players_keypoints is None: #Mapa 2D vacía
        fig.canvas.draw()
        rgba_buffer = np.array(fig.canvas.buffer_rgba())
        pitch_img = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return pitch_img


    if team_segmentation is None: #No team segmentation
        num_players = len(players_keypoints)
        team_segmentation = np.zeros(num_players)


    for (kpt,team) in zip(players_keypoints,team_segmentation):

    
        x_coords = kpt[0]
        y_coords = kpt[1]

        if team == 0:
            ax.scatter(x_coords, y_coords, color="red", s=100, edgecolors="white", linewidths=1.5, zorder=5)
        else:
            ax.scatter(x_coords, y_coords, color="blue", s=100, edgecolors="white", linewidths=1.5, zorder=5)


    fig.canvas.draw()
    rgba_buffer = np.array(fig.canvas.buffer_rgba())
    pitch_img = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    return pitch_img

def overlay_pitch_minimap(frame, pitch_img, alpha=0.7, scale_factor=0.35):
   
    #Ajustar el tamaño del minimapa proporcionalmente al frame
    frame_h, frame_w = frame.shape[:2]
    map_w = int(frame_w * scale_factor)
    aspect_ratio = pitch_img.shape[0] / pitch_img.shape[1]
    map_h = int(map_w * aspect_ratio)
    
    minimap = cv2.resize(pitch_img, (map_w, map_h), interpolation=cv2.INTER_AREA)

    # Abajo al Centro
    x_offset = (frame_w - map_w) // 2
    y_offset = frame_h - map_h - 10

    #Definir la Región de Interés (ROI) en el frame original
    roi = frame[y_offset:y_offset+map_h, x_offset:x_offset+map_w]

    # 4. Mezclar las imágenes para dar transparencia
    # Formula: frame_final = roi * (1 - alpha) + minimap * alpha
    blended = cv2.addWeighted(roi, 1 - alpha, minimap, alpha, 0)

    # 5. Insertar la mezcla de vuelta en el frame original
    frame_copy = frame.copy()
    frame_copy[y_offset:y_offset+map_h, x_offset:x_offset+map_w] = blended

    return frame_copy


def draw_keypoints(frame,keypoints:torch.Tensor, conf = 0.85):
    """
    Dibuja keypoints sobre la imagen dada
    """

    annotated_frame = frame.copy()

    if len(keypoints) > 1:
        keypoints = keypoints[0]

    keypoints = keypoints.squeeze(0)

    
    for i, kpt in enumerate(keypoints):

        x , y, kpt_conf = kpt
        if kpt_conf >= conf:
            cv2.circle(annotated_frame, (int(x), int(y)), radius=5, color=PALETA_COLORES[i], thickness=cv2.FILLED)
    
    return annotated_frame
    
