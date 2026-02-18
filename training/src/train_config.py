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

MODEL_FAMILY = "yolo11"
MODEL_SIZE = "n"


# ============================================================================
# Training Settings
# ============================================================================

EPOCHS = 80
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cpu"
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

PROJECT_DIR = "mlruns"


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
HSV_S = 0.6    # Saturation
HSV_V = 0.35    # Value/Brightness

# Geometric augmentation
DEGREES = 5.0      # Rotation
TRANSLATE = 0.05    # Translation
SCALE = 0.2        # Scale
SHEAR = 0.0        # Shear
PERSPECTIVE = 0.0  # Perspective

# Flip augmentation
FLIPUD = 0.0  # Vertical flip
FLIPLR = 0.5  # Horizontal flip

# Advanced augmentation
MOSAIC = 0      # Mosaic
MIXUP = 0       # MixUp
COPY_PASTE = 0.0  # Copy-paste


# ============================================================================
# Other Settings
# ============================================================================

AMP = True         # Automatic Mixed Precision
CACHE = False      # Cache images in RAM


# ============================================================================
# Presets
# ============================================================================

PRESET_QUICK_TEST = {
    "epochs": 5,
    "imgsz": 320,
    "batch": 8,
    "run_name": "quick_test",
}

PRESET_BASELINE = {
    "epochs": 80,
    "imgsz": 640,
    "batch": 16,
    "patience": 20,
    "run_name": "baseline",
}

PRESET_HEAVY_AUG = {
    "epochs": 150,
    "imgsz": 640,
    "batch": 32,
    "run_name": "heavy_augmentation",
}
