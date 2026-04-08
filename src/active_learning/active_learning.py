"""
Contiene la clase llamada LS
Permite, mediante la API, interactuar con Label Studio
"""

from dotenv import load_dotenv
import os
from label_studio_sdk import LabelStudio
import requests
from PIL import Image
import requests
from tqdm import tqdm
from io import BytesIO
import zipfile
import io
import shutil

load_dotenv()

class LS:

    def __init__(self,project_name,model,model_name):
        """
        Conexion al proyecto
        """
        self.model = model
        self.model_name = model_name
        #Datos de acceso
         
        self.API_KEY = os.getenv("API_KEY")
        self.LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        self.URL_REFRESH = os.getenv("URL_REFRESH") #Url para refrescar token de acceso

        self.IMG_PATH = os.getenv("IMG_PATH") #Ruta a las imagenes generadas
        self.COMPLETED_TASKS_PATH = os.getenv("COMPLETED_TASKS_PATH") #Ruta a la carpeta de imagenes completadas
        self.DATASET = os.getenv("DATASET")
        response = requests.post(self.URL_REFRESH,json={"refresh": self.API_KEY})
        self.ACCESS_TOKEN = response.json().get("access")

        #Conexion al proyecto
        #Conectamos con label studio
        self.client = LabelStudio(api_key=self.API_KEY)
        projects_info = self.client.projects.list()
        self.project = next((p for p in projects_info if p.title == project_name),None)
        self.project = self.client.projects.get(self.project.id)

    def refresh_token(self):
        response = requests.post(self.URL_REFRESH,json={"refresh": self.API_KEY})
        self.ACCESS_TOKEN = response.json().get("access")


    def delete_task(self,t_id):
        """
        Elimina el task asociado a t_id y la imagen asociada
        Args:
            t_id (int): id del task
        """
        self.refresh_token()
        task = self.client.tasks.get(id = t_id)
        self.client.tasks.delete(id=t_id)
        img_path = f'/{task.data["image"].split("=")[1]}'

        if os.path.exists(img_path):
            os.remove(img_path)


    def import_imgs(self):
        """
        Crea nuevas tareas asociadas a las nuevas imagenes generadas
        """
        storages = self.client.import_storage.local.list(project=self.project.id)

        for strg in storages:
            
            self.client.import_storage.local.sync(strg.id,request_options={"timeout_in_seconds": 7200})


    def sync_imgs(self):
        """
        Sincroniza las tareas con las imagenes almacenadas en IMG_PATH

        Comprueba todas las tareas del proyecto y elimina aquellas cuya imagen ya no existe en la carpeta del almacenamiento.Esto asegura
        que las tareas estén consistentes con el contenido actual del directorio de imagenes.
        """

        tasks= self.client.tasks.list(project=self.project.id)
        
        for i, task in enumerate(tqdm(tasks)):
            task_dict = task.dict()
            img_path = f'/{task_dict['data']['image'].split("=")[1]}'

            if not os.path.exists(img_path):
                self.delete_task(t_id=int(task_dict["id"]))
        

    def predict_image(self,image,img_width,img_height):
        """
        Realiza predicciones sobre una imagen usando el modelo.

        Args:
            image: imagen
            img_width : anchura imagen
            img_height: altura imagen

        Returns:
            list of dict: Lista de predicciones, cada una con la siguiente estructura:
        
            - 'result' (list of dict): Lista de objetos detectados, cada uno con:
                - 'from_name' (str): Nombre de la etiqueta de origen ('label').
                - 'to_name' (str): Nombre del destino ('img').
                - 'original_width' (int): Ancho de la imagen original.
                - 'original_height' (int): Alto de la imagen original.
                - 'image_rotation' (int): Rotación de la imagen (0 si no hay rotación).
                - 'value' (dict): Contiene las coordenadas y etiquetas del objeto detectado:
                    - 'rotation' (int): Rotación del objeto (0 si no hay).
                    - 'rectanglelabels' (list of str): Lista con la etiqueta del objeto.
                    - 'width' (float): Ancho del rectángulo como porcentaje de la imagen.
                    - 'height' (float): Alto del rectángulo como porcentaje de la imagen.
                    - 'x' (float): Coordenada X del centro del rectángulo como porcentaje de la imagen.
                    - 'y' (float): Coordenada Y del centro del rectángulo como porcentaje de la imagen.
                - 'score' (float): Confianza de la predicción.
                - 'type' (str): Tipo de objeto ('rectanglelabels').
            - 'score' (float): Score global de la predicción (mínimo de los scores individuales).
            - 'model_version' (str): Nombre o versión del modelo utilizado.

        """

        results = self.model(image,verbose = False)
        predictions = []
        for result in results:
            
            boxes = result.boxes.cpu().numpy()
            prediction = {'result': [], 'score': 0.0, 'model_version': self.model_name}
            scores = []
            for box, class_id, score in zip(boxes.xywh, boxes.cls, boxes.conf):
                x, y, w, h = box
                prediction['result'].append({
                    'from_name': 'label',
                    'to_name': 'img',
                    'original_width': int(img_width),
                    'original_height': int(img_height),
                    'image_rotation': 0,
                    'value': {
                        'rotation': 0,
                        'rectanglelabels': [result.names[class_id]],
                        'width': float(w / img_width * 100),
                        'height': float(h / img_height * 100),
                        'x': float((x - 0.5 * w) / img_width * 100),
                        'y': float((y - 0.5 * h) / img_height * 100)
                    },
                    'score': float(score),
                    'type': 'rectanglelabels',
                })
                scores.append(float(score))
            prediction['score'] = min(scores) if scores else 0.0
            predictions.append(prediction)

        return predictions
    
    def annotate_image(self,t_id):

        """
        Pasa por el modelo la tarea asociada a t_id y crea los correspondientes bounding boxes

        Args:
            t_id (int) : id del task

        """

        task = self.client.tasks.get(id = t_id)
        task_properties = task.dict()
        img_path = task_properties['data']['image']
         
        img= '/'+img_path[img_path.find('home/'):]
        if os.path.exists(img): #Comprueba que esta la imagen
            url = f'{self.LABEL_STUDIO_URL}{task.data['image']}'
            self.refresh_token()
            request = requests.get(url, headers={'Authorization': f'Bearer {self.ACCESS_TOKEN}'}, stream=True)
            image = Image.open(request.raw)
            w,h = image.size
            predictions = self.predict_image(image,w,h)[0]
            self.client.predictions.create(task=task.id, result=predictions['result'], score=float(predictions['score']), model_version=predictions['model_version'])


    def predict_tasks(self):
        """
        Pasa las imagenes NO COMPLETADAS por el modelo   
        """
       
        tasks = self.client.tasks.list(project=self.project.id)
        for i, task in enumerate(tqdm(tasks)):
            t_properties = task.dict()

            if t_properties['completed_at'] is None: #Solo predecir sobre imagens que no esten completadas
                self.annotate_image(t_id= int(t_properties['id']))

   

    def export_completed_tasks(self):
        """
        Exporta las imagenes y las etiquetadas de las tareas COMPLETADAS al  la carpeta 
        COMPLETED_TASKs
        """

        #Creamos una nueva snapshot
        url =f'{self.LABEL_STUDIO_URL}/api/projects/{self.project.id}/exports/'
        self.refresh_token()
        snapshot_rq = requests.post(url,headers={'Authorization': f'Bearer {self.ACCESS_TOKEN}'},json = {"task_filter_options": {"finished": "only"}})

        if snapshot_rq.status_code == 201:
            data = snapshot_rq.json()
            export_pk = data["id"]
        else:
            print(snapshot_rq.text)

        #Descargamos la snapshot en formato YOLO
        url = f'{self.LABEL_STUDIO_URL}/api/projects/{self.project.id}/exports/{export_pk}/download?exportType=YOLO'
        self.refresh_token()
        download_snapshot_rq = requests.get(url,headers={'Authorization': f'Bearer {self.ACCESS_TOKEN}'})

        if download_snapshot_rq.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(download_snapshot_rq.content)) as z:
            # Extraemos todo en la carpeta actual (o especifica otra ruta)
                z.extractall(path=self.COMPLETED_TASKS_PATH)
                print("Donwload ok")

            #Copiamos las imagenes y la etiquetas a fine_tuning
            labels_path = os.path.join(self.COMPLETED_TASKS_PATH,"labels")
            imgs_path = os.path.join(self.IMG_PATH)
            labels = os.listdir(labels_path)

            for label in tqdm(labels):
                name,extension = os.path.splitext(label)

                #Rutas originales
                img_path = os.path.join(imgs_path,name+".jpg")
                label_path = os.path.join(labels_path,label)

                #Mover a rutas destinos
                try:
                    shutil.copy(img_path,os.path.join(self.COMPLETED_TASKS_PATH,"images",name+".jpg"))
                except:
                    print(f"Error during copy: {name}.jpg")


    def move2dataset(self):
        """
        Copia las imagenes y las etiquetas de las tareas COMPLETADAS al dataset base.

        Las imagenes se añaden a la carpeta Train del Dataset
        """
        imgs_path = os.path.join(self.COMPLETED_TASKS_PATH,"images")
        labels_path = os.path.join(self.COMPLETED_TASKS_PATH,"labels")
        imgs = os.listdir(imgs_path)

        for img in tqdm(imgs):
            name,_ = os.path.splitext(img) #Separar nombre de extension

            shutil.copy(os.path.join(imgs_path,name+".jpg"), 
                        os.path.join(self.DATASET,"images",name+".jpg")) #Copioamos la imagen
            
            shutil.copy(os.path.join(labels_path,name+".txt"), 
                        os.path.join(self.DATASET,"labels",name+".txt")) #Copioamos la etiqueta
                        
        
  
                

        
    

    



        

        
        


    