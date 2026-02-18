"""
Training Configuration File
Customize these settings for your YOLO training
"""

# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET_YAML = "prepared_data/dataset.yaml"


# ============================================================================
# Model Configuration
# ============================================================================

MODEL_FAMILY = "yolo11"  # "yolo11" or "yolo26"
MODEL_SIZE = "n"         # "n", "s", "m", "l", "x"


# ============================================================================
# Training Settings
# ============================================================================

EPOCHS = 80
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cpu"  # "0" for GPU, "cpu" for CPU, "mps" for Mac
WORKERS = 8
SEED = 42


# ============================================================================
# MLflow Settings
# ============================================================================

MLFLOW_EXPERIMENT = "yolo_training"
RUN_NAME = "yolo11n_baseline"


# ============================================================================
# Output Settings
# ============================================================================

PROJECT_DIR = "runs"  # Directory where Ultralytics saves results


# ============================================================================
# Optimizer Hyperparameters
# ============================================================================

LEARNING_RATE_INITIAL = 0.01  # lr0
LEARNING_RATE_FINAL = 0.01    # lrf
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
PATIENCE = 30  # Early stopping patience


# ============================================================================
# Data Augmentation
# ============================================================================

# HSV augmentation
HSV_H = 0.015  # Hue
HSV_S = 0.7    # Saturation
HSV_V = 0.4    # Value/Brightness

# Geometric augmentation
DEGREES = 0.0      # Rotation
TRANSLATE = 0.1    # Translation
SCALE = 0.5        # Scale
SHEAR = 0.0        # Shear
PERSPECTIVE = 0.0  # Perspective

# Flip augmentation
FLIPUD = 0.0  # Vertical flip
FLIPLR = 0.5  # Horizontal flip

# Advanced augmentation
MOSAIC = 1.0      # Mosaic (4 images)
MIXUP = 0.1       # MixUp (2 images)
COPY_PASTE = 0.0  # Copy-paste


# ============================================================================
# Other Settings
# ============================================================================

AMP = True         # Automatic Mixed Precision
CACHE = False      # Cache images in RAM (faster but uses more memory)


# ============================================================================
# Presets (Quick configurations)
# ============================================================================

PRESET_QUICK_TEST = {
    "epochs": 5,
    "imgsz": 320,
    "batch": 8,
    "run_name": "quick_test",
}

PRESET_BASELINE = {
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "run_name": "baseline",
    "translate": 0.05,
    "scale": 0.3,
    "fliplr": 0.3,
    "hsv_h": 0.02,
    "hsv_v": 0.5,
}

PRESET_HEAVY_AUG = {
    "epochs": 150,
    "model_size": "s",
    "hsv_h": 0.03,
    "hsv_s": 0.9,
    "hsv_v": 0.6,
    "degrees": 15.0,
    "translate": 0.2,
    "scale": 0.9,
    "mixup": 0.3,
    "run_name": "heavy_augmentation",
}

PRESET_PRODUCTION = {
    "epochs": 200,
    "model_size": "m",
    "batch": 32,
    "patience": 50,
    "run_name": "production",
}