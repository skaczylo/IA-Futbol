from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- DATASETS ---
DATASETS_DIR = PROJECT_ROOT / "datasets"
OBJ_DETECTION_DATA = DATASETS_DIR / "object_detection"
PITCH_KEYPOINTS_DATA = DATASETS_DIR / "pitch_keypoints"
TEAM_SEGMENTATION_DATA = DATASETS_DIR / "team_segmentation"

# --- RUNS / EXPERIMENTS (Modelos y Resultados) ---
RUNS_DIR = PROJECT_ROOT / "runs"

# Carpetas específicas para guardar los resultados de cada tarea
OBJ_DETECTION_RUNS = RUNS_DIR / "object_detection"
PITCH_KEYPOINTS_RUNS = RUNS_DIR / "pitch_keypoints"
TEAM_SEGMENTATION_RUNS = RUNS_DIR / "team_segmentation"