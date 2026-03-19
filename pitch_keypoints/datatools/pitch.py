



def overlay_pitch(background, overlay, padding=20, alpha=0.5, target_scale=0.25):
    """
    Superpone la imagen del campo (overlay) en la principal (background),
    centrada-abajo, con transparencia y reescalada a un porcentaje
    específico del tamaño total.

    Parámetros:
    -----------
    - background: El frame o imagen de fondo (numpy array).
    - overlay: La imagen del campo 2D a superponer.
    - padding: Margen desde el borde inferior.
    - alpha: Opacidad de la imagen del campo (0.0 a 1.0).
    - target_scale: Porcentaje del ancho total que debe ocupar el campo.
                    (0.10 para 10%, 0.15 para 15%, etc.) (Defecto: 0.25)
    """
    # Evitar errores si una de las imágenes está vacía
    if background is None or overlay is None:
        return background

    #dimensiones originales
    h_bg, w_bg = background.shape[:2]
    h_ov, w_ov = overlay.shape[:2]


    # Calcular el nuevo ancho objetivo
    new_w = int(w_bg * target_scale)
    new_h = int((new_w * h_ov) / w_ov)
    

    # Redimensionar la imagen del campo
    overlay_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h_ov, w_ov = overlay_resized.shape[:2] # Actualizar dimensiones finales

    # 2. Calcular coordenadas (Centro-Abajo)
    x_offset = (w_bg // 2) - (w_ov // 2)
    y_offset = h_bg - h_ov - padding


    # Región de Interés en la imagen de fondo
    roi = background[y_offset:y_offset+h_ov, x_offset:x_offset+w_ov]

    # Transparencia
    beta = 1.0 - alpha
    blended_roi = cv2.addWeighted(overlay_resized, alpha, roi, beta, 0)

    combined = background.copy() 
    combined[y_offset:y_offset+h_ov, x_offset:x_offset+w_ov] = blended_roi

    return combined





















