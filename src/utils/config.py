from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- DATASETS ---
DATASETS_DIR = PROJECT_ROOT / "datasets"
OBJ_DETECTION_DATA = DATASETS_DIR / "object_detection"
PITCH_KEYPOINTS_DATA = DATASETS_DIR / "pitch_keypoints"
DATA_GENERATED_IMGS = DATASETS_DIR / "data_generated" / "images"


# --- VIDEOS ---
VIDEOS_DIR = PROJECT_ROOT/ "videos"

# --- IMAGENES ---
IMAGES_DIR = PROJECT_ROOT / "images"

# --- RUNS / EXPERIMENTS (Modelos y Resultados) ---
RUNS_DIR = PROJECT_ROOT / "runs"


# Carpetas específicas para guardar los resultados de cada tarea
OBJ_DETECTION_RUNS = RUNS_DIR / "object_detection"
OBJ_DETECTION_RUN0 = OBJ_DETECTION_RUNS/"it_0"
OBJ_DETECTION_RUN1 = OBJ_DETECTION_RUNS/"it_1"
OBJ_DETECTION_RUN2 = OBJ_DETECTION_RUNS/"it_2"


BEST_OBJ_DETECTION_MODEL = OBJ_DETECTION_RUNS / "it_2" / "weights" / "best.pt"


PITCH_KEYPOINTS_RUNS = RUNS_DIR / "pitch_keypoints"
BEST_KPT_MODEL = PITCH_KEYPOINTS_RUNS / "train" / "weights" / "best.pt"

#ENV
ENV_PATH = PROJECT_ROOT / ".env"
