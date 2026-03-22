import os
from ultralytics import YOLO
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from umap import UMAP
from sklearn.cluster import KMeans

#CONSTANTES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

# transformaciones para las imágenes (DINO requiere 224x224 y normalización)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_crops(img,object_detection_model):
    """
    Dada una imagen extrae recortes de jugadores
    """
    detections = object_detection_model(img,verbose = False)[0]

    clss = detections.boxes.cls

    crops = []
    for i, box in enumerate(detections.boxes.xywh):

        if clss[i] == 1: #es jugador

            x, y, w, h = box
           
            x1 = x - w/2
            x2 = x + w/2
            y1 = y - h/2
            y2 = y + h/2

            crop = img[int(y1):int(y2), int(x1):int(x2)]

            crops.append(crop)

    return crops  


def get_dino_embeddings(crops):
    """
    Crops: array de imagenes
    """
    embeddings = []
    for crop in crops:
        
        # Convertir de BGR (OpenCV) a RGB y a PIL Image
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(img_tensor)
            embeddings.append(embedding.cpu().numpy().flatten())
            
    return np.array(embeddings)