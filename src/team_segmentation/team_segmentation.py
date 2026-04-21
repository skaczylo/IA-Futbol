import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from umap import UMAP
from sklearn.cluster import KMeans
import supervision as sv
from ultralytics import YOLO


class TeamSegmentation:
    def __init__(self, n_clusters=2, n_neighbors=15, random_state=42):
        """
        Inicializa la clase cargando los modelos necesarios (DINOv2) y 
        preparando los algoritmos de clustering (UMAP y KMeans).
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    
        #DinoV2
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.encoder.eval()
        
        #Configurar transformaciones de imagen
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Modelos de clasificación
        self.reducer = UMAP(n_components=3, n_neighbors=n_neighbors, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        
        # Estado interno
        self.is_calibrated = False

    def _crop_detections(self, image: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
        """Extrae los recortes de los jugadores asegurando que no se salgan de la imagen."""
        crops = []
        h_img, w_img = image.shape[:2]
        
        for bbox in detections.xyxy:
            x_min, y_min, x_max, y_max = map(int, bbox)
            
            # Limitar a los bordes de la imagen
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w_img, x_max), min(h_img, y_max)
            
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size > 0:
                crops.append(crop)
                
        return crops

   

    def _encode_crops(self, crops, batch_size=32) -> np.ndarray:
        """Convierte las imágenes recortadas en vectores usando DINOv2 en lotes (batches)."""

        if not crops:
            return np.array([])

        tensors = []
        for crop in crops:
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            # No usamos unsqueeze(0) aquí. Solo guardamos el tensor [C, H, W]
            tensors.append(self.transform(img_pil))
            
        all_embeddings = []
        
        for i in range(0, len(tensors), batch_size):
            
            batch_list = tensors[i:i + batch_size]
            batch_tensor = torch.stack(batch_list).to(self.device)
            
            with torch.no_grad():
               
                embeddings = self.encoder(batch_tensor)
                all_embeddings.append(embeddings.cpu().numpy())
                

        return np.vstack(all_embeddings)

    def preprocess_from_video(self, video_path, object_detector, max_crops=400):
        """
        Lee una muestra del vídeo, extrae jugadores y entrena UMAP y KMeans.
        """
    
        cap = cv2.VideoCapture(video_path)
        crops_muestra = []
        frame_count = 0
        
        while cap.isOpened() and len(crops_muestra) <= max_crops:
            ret, frame = cap.read()
            if not ret:
                break
                
        
            if frame_count % 50 == 0:
                results = object_detector(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[detections.class_id == 1] # Solo personas
                
                crops = self._crop_detections(frame, detections)
                crops_muestra.extend(crops)
                
            frame_count += 1
            
        cap.release()
        
        
        # Obtener características y entrenar modelos
        features = self._encode_crops(crops_muestra)
        features_3d = self.reducer.fit_transform(features)
        self.kmeans.fit(features_3d)
        
        self.is_calibrated = True
        

    def calculate_target_positions(self, H, players: sv.Detections) -> tuple[sv.Detections, sv.Detections]:

        """
        Calcula las posiciones objetivo de los jugadores en el minimapa usando la homografía.

        Devuelve una lista con las coordenadas de los jugadores proyectados en el minimapa
        """
        if (len(players) == 0):
            return np.array([])
        
        
        bottom_players= players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) 
        bottom_players = np.array(bottom_players, dtype=np.float32).reshape(-1, 1, 2) #(N,2) -> (N,1,2) para cv2

        bottom_players= players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) 
        bottom_players = np.array(bottom_players, dtype=np.float32).reshape(-1, 1, 2) #(N,2) -> (N,1,2) para cv2
        bottom_players_2dmap = cv2.perspectiveTransform(bottom_players, H)
        bottom_players_2dmap = bottom_players_2dmap.reshape(-1, 2)

        return bottom_players_2dmap

    def predict_teams(self, frame: np.ndarray, detections: sv.Detections) -> tuple[sv.Detections, sv.Detections]:
        """
        Asigna un equipo a cada jugador detectado en el frame.
        
        Devuelve:
            - equipo_0 (sv.Detections): Objeto con los jugadores del primer equipo.
            - equipo_1 (sv.Detections): Objeto con los jugadores del segundo equipo.
        """
        
        if not self.is_calibrated:
            raise RuntimeError("Debes llamar a 'calibrate_from_video' antes de predecir.")
            
       
        if len(detections) == 0:
            return sv.Detections.empty(), sv.Detections.empty()
            
            
        crops = self._crop_detections(frame, detections)
        
        if crops:
            features = self._encode_crops(crops)
            
            # Inferencia rápida (transform y predict)
            features_3d = self.reducer.transform(features)
            labels = self.kmeans.predict(features_3d)
            
            # Sobrescribimos el class_id de TODOS los jugadores con el id de su equipo (0 o 1)
            detections.class_id = labels
            
            # --- AQUÍ ESTÁ LA MAGIA DE SUPERVISION ---
            # Filtramos el objeto original para crear dos nuevos objetos sv.Detections
            equipo_0 = detections[detections.class_id == 0]
            equipo_1 = detections[detections.class_id == 1]
            
            return equipo_0, equipo_1
            
        # Por seguridad, si fallan los recortes
        return sv.Detections.empty(), sv.Detections.empty()