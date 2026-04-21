import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from matplotlib.patches import Arc, Circle, Rectangle
import numpy as np
import cv2
from .image_tools import fig_to_image
import supervision as sv
PALETA_COLORES = np.random.randint(0, 256, size=(60, 3), dtype=int).tolist()
GREEN = '#4C9A2A'


def draw_pitch(figsize=(12, 8)):
    # Colores tipo "Premium"
   
    GREEN_DARK = "#224422"
    GREEN_LIGHT = "#2b5329"
    LINE_COLOR = "white"
    
    # Creamos la figura eliminando márgenes internos de Matplotlib
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Medidas oficiales FIFA
    pitch_length, pitch_width = 105.0, 68.0
    half_len, half_wid = pitch_length / 2, pitch_width / 2
    
    # 1. DIBUJAR EL CÉSPED (Franjas verticales)
    num_stripes = 10
    stripe_width = pitch_length / num_stripes
    for i in range(num_stripes):
        color = GREEN_LIGHT if i % 2 == 0 else GREEN_DARK
        ax.add_patch(Rectangle((-half_len + i*stripe_width, -half_wid), 
                               stripe_width, pitch_width, 
                               fill=True, color=color, zorder=0))

    # 2. LÍNEAS PERIMETRALES Y MEDIO CAMPO
    ax.plot([-half_len, -half_len, half_len, half_len, -half_len], 
            [-half_wid, half_wid, half_wid, -half_wid, -half_wid], 
            color=LINE_COLOR, linewidth=2, zorder=3)
    ax.plot([0, 0], [-half_wid, half_wid], color=LINE_COLOR, linewidth=2, zorder=3)
    
    ax.add_patch(Circle((0, 0), 9.15, color=LINE_COLOR, fill=False, linewidth=2, zorder=3))
    ax.plot(0, 0, "o", color=LINE_COLOR, markersize=5, zorder=3)

    # 3. ÁREAS (Penalti y Meta)
    for side in [-1, 1]:
        # Área Grande
        ax.add_patch(Rectangle((side * half_len, -20.16), -side * 16.5, 40.32, 
                               fill=False, edgecolor=LINE_COLOR, linewidth=2, zorder=3))
        # Área Chica
        ax.add_patch(Rectangle((side * half_len, -9.16), -side * 5.5, 18.32, 
                               fill=False, edgecolor=LINE_COLOR, linewidth=2, zorder=3))
        # Punto de penalti
        ax.plot(side * (half_len - 11), 0, "o", color=LINE_COLOR, markersize=4, zorder=3)
        # Arcos de penalti
        theta_1, theta_2 = (127, 233) if side == 1 else (-53, 53)
        ax.add_patch(Arc((side * (half_len - 11), 0), 18.3, 18.3, theta1=theta_1, theta2=theta_2, 
                         color=LINE_COLOR, linewidth=2, zorder=3))

    # 4. PORTERÍAS (7.32m de ancho, sobresalen 2m del campo)
    for side in [-1, 1]:
        ax.plot([side * half_len, side * (half_len + 2), side * (half_len + 2), side * half_len], 
                [-3.66, -3.66, 3.66, 3.66], color=LINE_COLOR, linewidth=2, zorder=1)

    # --- AJUSTES DE MARCO ---
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Ajustamos los límites casi al ras de las porterías
    # Esto elimina el "aire" innecesario alrededor
    ax.set_xlim(-half_len - 1, half_len + 1)
    ax.set_ylim(-half_wid - 1, half_wid + 1)
    
    # Forzar a que el dibujo ocupe toda la ventana del plot
    plt.close(fig)
    
    return fig, ax


def draw_objects_on_pitch(pitch, players, color = "#DF0A0A", object_size=20):
    """
    Dibuja objetos (jugadores, balón, etc.) sobre el mapa del campo.
    
    Args:
        pitch: Tupla (fig, ax) devuelta por draw_pitch().
        players: Lista o array de coordenadas [(x1, y1), (x2, y2), ...].
        colors: Lista de colores strings o RGB para cada objeto.
        object_size: Tamaño del marcador (puntos).
    """
    fig, ax = pitch

    if len(players) == 0:
        return fig, ax
    

    ax.invert_yaxis()

    # Iteramos sobre las coordenadas y los colores simultáneamente
    for player in players:
        x,y = player
        
        # Dibujamos el punto en el campo
        ax.plot(x, y, "o", 
                color=color, 
                markersize=object_size, 
                markeredgecolor="white", # Borde blanco para resaltar
                markeredgewidth=0.5,
                zorder=10)
    return fig, ax



def overlay_pitch_on_image(image, pitch, scale=0.25, padding_bottom=20, alpha=0.5):
    """
    Superpone el dibujo del campo (minimapa) sobre la imagen principal con transparencia.
    
    Args:
        image: Imagen original del video (numpy array BGR).
        pitch: Tupla (fig, ax) devuelta por draw_pitch().
        scale: Porcentaje del ancho de la imagen principal (ej. 0.20 = 20%).
        padding_bottom: Margen en píxeles desde el borde inferior.
        alpha: Opacidad del minimapa (1.0 = totalmente opaco, 0.0 = invisible).
        
    Returns:
        np.ndarray: La imagen con el minimapa semitransparente superpuesto.
    """
  
    pitch_img = fig_to_image(pitch)
    
    # 2. Obtener dimensiones
    h_main, w_main = image.shape[:2]
    h_pitch, w_pitch = pitch_img.shape[:2]
    
    # 3. Calcular el nuevo tamaño
    new_w = int(w_main * scale)
    aspect_ratio = h_pitch / w_pitch
    new_h = int(new_w * aspect_ratio)
    
    pitch_resized = cv2.resize(pitch_img, (new_w, new_h))
    
    # 4. Calcular coordenadas (Abajo al Centro)
    x_start = (w_main - new_w) // 2
    x_end = x_start + new_w
    
    y_start = h_main - new_h - padding_bottom
    y_end = y_start + new_h
    
    if y_start < 0:
        y_start = 0
        y_end = new_h
        
    # --- AQUÍ ESTÁ LA MAGIA DE LA TRANSPARENCIA ---
    
    # 5. Extraemos la Región de Interés (ROI) de la imagen de fondo
    roi = image[y_start:y_end, x_start:x_end]
    
    # 6. Mezclamos ambas imágenes
    # Fórmula: (pitch * alpha) + (roi * (1 - alpha)) + 0
    blended = cv2.addWeighted(pitch_resized, alpha, roi, 1 - alpha, 0)
    
    # 7. Volvemos a colocar la imagen mezclada en el frame original
    image[y_start:y_end, x_start:x_end] = blended
    
    return image


# Ejemplo de uso:
if __name__ == "__main__":
    fig, ax = draw_pitch()
    plt.show()