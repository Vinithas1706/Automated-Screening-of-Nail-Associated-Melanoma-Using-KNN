"""Central configuration shared by training, evaluation, prediction and the Flask app."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Folders
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"          # json / csv outputs
CHARTS_DIR = BASE_DIR / "static" / "charts"  # png charts shown on the web app
UPLOAD_DIR = BASE_DIR / "uploads"            # NOT inside static/: served only to the logged-in owner
INSTANCE_DIR = BASE_DIR / "instance"
DB_PATH = INSTANCE_DIR / "users.db"           # SQLite user accounts
SECRET_KEY_PATH = INSTANCE_DIR / "secret.key"

# Files
MODEL_PATH = MODELS_DIR / "knn_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
GRID_PATH = REPORTS_DIR / "knn_grid_results.csv"
AUDIT_PATH = REPORTS_DIR / "dataset_audit.json"
CLEAN_LIST_PATH = REPORTS_DIR / "clean_dataset_list.csv"
SUBJECTS_CSV = DATASET_DIR / "subjects.csv"   # optional: filename,subject_id

# Classes: folder name -> numeric label (melanoma is the "positive" class)
CLASS_FOLDERS = {"non_melanoma": 0, "melanoma": 1}
CLASS_NAMES = {0: "Non-Melanoma", 1: "Melanoma"}

# Image handling
IMG_SIZE = 224
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGE_SIDE = 100          # reject tiny uploads
MAX_UPLOAD_MB = 5

# Experiment settings
RANDOM_STATE = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / validation / test
K_VALUES = list(range(1, 32, 2))
DISTANCE_METRICS = ["euclidean", "manhattan", "chebyshev", "cosine"]
WEIGHTINGS = ["uniform", "distance"]
