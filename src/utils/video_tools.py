import cv2
from tqdm import tqdm

def process_and_write_video(input_path: str, output_path: str, process_frame_func:callable, **kwargs):
    """
    Lee un video, aplica una función de procesamiento a cada frame y guarda el resultado.
    
    Args:
        input_path (str): Ruta del video original.
        output_path (str): Ruta donde se guardará el nuevo video.
        process_frame_func (callable): Función que recibe un frame (y kwargs) y devuelve el frame modificado.
        **kwargs: Argumentos extra que pueda necesitar tu función de procesamiento (modelos, config, etc).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Error al abrir el video: {input_path}")

    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurar el writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    with tqdm(total=total_frames, desc="Procesando video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- AQUÍ OCURRE LA MAGIA ---
            # Llamamos a tu función personalizada pasándole el frame actual
            # y cualquier otro modelo o dato que necesite a través de kwargs
            processed_frame = process_frame_func(frame, frame_idx=frame_idx, **kwargs)
            
            writer.write(processed_frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()