from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from utils.config import BEST_OBJ_DETECTION_MODEL, ENV_PATH, DATA_GENERATED_IMGS

load_dotenv(ENV_PATH)

LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL', 'http://localhost:8080')
LABEL_STUDIO_TOKEN = os.getenv('LABEL_STUDIO_ACCESS_TOKEN')

LABEL_MAP = {
    'ball': 'Ball',
    'player': 'Player',
    'referee': 'Referee',
}

class YOLOBackend(LabelStudioMLBase):
    _model = None  # ← compartido, solo se carga una vez

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if YOLOBackend._model is None:
            print("=== CARGANDO MODELO (primera vez) ===")
            YOLOBackend._model = YOLO(BEST_OBJ_DETECTION_MODEL)
            YOLOBackend._model.to('cpu')
            print("=== MODELO CARGADO ===")
        self.model = YOLOBackend._model
        self.model_name = str(BEST_OBJ_DETECTION_MODEL)

    def predict(self, tasks, context=None, **kwargs):
        predictions = []

        for task in tasks:
            image_url = task['data'].get('image')
            image, img_width, img_height = self._load_image(image_url)
            if image is None:
                predictions.append({'result': [], 'score': 0.0})
                continue

            results = self.model(image, verbose=False, device='cpu', imgsz=640)

            result_list = []
            scores = []

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box, class_id, score in zip(boxes.xywh, boxes.cls, boxes.conf):
                    x, y, w, h = box
                    label = result.names[int(class_id)]
                    label_mapped = LABEL_MAP.get(label, label)

                    result_list.append({
                        'from_name': 'label',
                        'to_name': 'image',
                        'original_width': int(img_width),
                        'original_height': int(img_height),
                        'image_rotation': 0,
                        'value': {
                            'rotation': 0,
                            'rectanglelabels': [label_mapped],
                            'width': float(w / img_width * 100),
                            'height': float(h / img_height * 100),
                            'x': float((x - 0.5 * w) / img_width * 100),
                            'y': float((y - 0.5 * h) / img_height * 100),
                        },
                        'score': float(score),
                        'type': 'rectanglelabels',
                    })
                    scores.append(float(score))

            predictions.append({
                'result': result_list,
                'score': min(scores) if scores else 0.0,
                'model_version': self.model_name,
            })

        return predictions

    def _load_image(self, image_url):
        try:
            if '/local-files/' in image_url:
                parsed = urlparse(image_url)
                relative_path = parse_qs(parsed.query).get('d', [None])[0]
                img_name = os.path.basename(relative_path)
                full_path = os.path.join(DATA_GENERATED_IMGS, img_name)
                image = Image.open(full_path).convert('RGB')

            elif image_url.startswith('http'):
                headers = {'Authorization': f'Token {LABEL_STUDIO_TOKEN}'}
                response = requests.get(image_url, headers=headers)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')

            else:
                image = Image.open(image_url).convert('RGB')

            img_width, img_height = image.size
            return image, img_width, img_height

        except Exception as e:
            print(f"[ERROR] No se pudo cargar la imagen: {image_url} — {e}")
            return None, None, None


if __name__ == "__main__":
    import tempfile
    from label_studio_ml.api import init_app
    app = init_app(
        model_class=YOLOBackend,
        model_dir=tempfile.mkdtemp()
    )
    app.run(host="0.0.0.0", port=9090, debug=True)