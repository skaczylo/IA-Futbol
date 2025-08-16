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
        self.DOWNLOAD_PATH = os.getenv("DOWNLOAD_PATH") #Ruta donde se descargan las imagenes completadas
        self.IMG_PATH = os.getenv("IMG_PATH") #Ruta a las imagenes generadas
        self.COMPLETED_TASKS_PATH = os.getenv("COMPLETED_TASKS_PATH") #Ruta a la carpeta Fine tuning
        self.DATASET = os.getenv("DATASET")
        response = requests.post(self.URL_REFRESH,json={"refresh": self.API_KEY})
        self.ACCESS_TOKEN = response.json().get("access")

        #Conexion al proyecto
        #Conectamos con label studio
        self.client = LabelStudio(api_key=self.API_KEY)
        projects_info = self.client.projects.list()
        self.project = next((p for p in projects_info if p.title == project_name),None)
        self.project = self.client.projects.get(self.project.id)

       
    def import_imgs(self):
        #sincronizamos datos
        storages = self.client.import_storage.local.list(project=self.project.id)

        for strg in storages:
            
            self.client.import_storage.local.sync(
                    strg.id,
                    request_options={"timeout_in_seconds": 7200}  # 2 horas
            )


    def refresh_token(self):
        response = requests.post(self.URL_REFRESH,json={"refresh": self.API_KEY})
        self.ACCESS_TOKEN = response.json().get("access")

    def predict_img(self,image,img_width,img_height):
        """
        Dada una imagen, el modelo del proyecto predice la imagen
        """
        results = self.model(image,verbose = False)
        predictions = []
        for result in results:
            img_width, img_height = result.orig_shape
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
    

    def predict_all_tasks(self):

        tasks = self.client.tasks.list(project=self.project.id)
        for i, task in enumerate(tqdm(tasks)):
        
            url = f'{self.LABEL_STUDIO_URL}{task.data['image']}'
           
            self.refresh_token()
            request = requests.get(url, headers={'Authorization': f'Bearer {self.ACCESS_TOKEN}'}, stream=True)
            image = Image.open(request.raw)
               
            w,h = image.size
            predictions = self.predict_img(image,w,h)[0]
            self.client.predictions.create(task=task.id, result=predictions['result'], score=float(predictions['score']), model_version=predictions['model_version'])

    def predict_new_tasks(self):

        tasks = self.client.tasks.list(project = self.project.id)

        for i, task in enumerate(tqdm(tasks)):
            dic = task.dict()

            if dic["total_predictions"] == 0:
                url = f'{self.LABEL_STUDIO_URL}{task.data['image']}'
                self.refresh_token()
                request = requests.get(url, headers={'Authorization': f'Bearer {self.ACCESS_TOKEN}'}, stream=True)
                image = Image.open(request.raw)
                w,h = image.size
                predictions = self.predict_img(image,w,h)
                self.client.predictions.create(task=task.id, result=predictions['result'], score=predictions['score'], model_version=predictions['model_version'])



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
                z.extractall(path=self.DOWNLOAD_PATH)
                print("Donwload ok")

            #Copiamos las imagenes y la etiquetas a fine_tuning
            labels_path = os.path.join(self.DOWNLOAD_PATH,"labels")
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
        Mueve las imagenes y las etiquetas de las tareas COMPLETADAS al dataset base

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
                        
        
    def delete_completed_tasks(self):
        """
        Elimina todos los tasks completados
        """

        tasks = self.client.tasks.list(project = self.project.id)
    
        for task in tasks:
        
            dic = task.dict()
            if dic["completed_at"] != None: #esta completada
                self.client.tasks.delete(id = dic["id"])#eliminar task
                os.remove(f'/{task.data["image"].split("=")[1]}') #eliminar del disco
                

        

    def delete_task(self,t_id):
        task = self.client.tasks.get(id = t_id)
        self.client.tasks.delete(id=t_id)
        os.remove(f'/{task.data["image"].split("=")[1]}')

    def delete_all_tasks(self):
        """
        Elimina todas las imagenes a marcar
        """
        self.client.tasks.delete_all_tasks(self.project.id)



        

        
        


    