import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from datatools.ellipse import PITCH_POINTS

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
    ax.set_aspect('equal') # Esto asegura que no se deformen los círculos si cambias el tamaño de la figura
    ax.axis('off')
    ax.set_xlim(-54.5, 54.5)
    ax.set_ylim(-35, 35)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig, ax
    
