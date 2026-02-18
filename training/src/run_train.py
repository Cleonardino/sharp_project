"""
Simple Training Runner
Uses configuration from train_config.py
"""

import training.src.train_config as cfg
from training.src.train import train_yolo, TrainingConfig


def run_training(preset: str = None, **overrides):
    """
    Run training with configuration from train_config.py
    
    Args:
        preset: Name of preset ("quick_test", "baseline", "heavy_aug", "production")
        **overrides: Any parameters to override
        
    Examples:
        # Default configuration
        run_training()
        
        # With preset
        run_training(preset="quick_test")
        
        # With overrides
        run_training(preset="baseline", epochs=100, device="0")
        
        # Fully custom
        run_training(device="cpu", epochs=50, batch=8)
    """
    
    # Load environment
    config = TrainingConfig()
    config.load_env()
    
    # Start with base configuration
    params = {
        "dataset_yaml": cfg.DATASET_YAML,
        "model_family": cfg.MODEL_FAMILY,
        "model_size": cfg.MODEL_SIZE,
        "epochs": cfg.EPOCHS,
        "imgsz": cfg.IMAGE_SIZE,
        "batch": cfg.BATCH_SIZE,
        "device": cfg.DEVICE,
        "workers": cfg.WORKERS,
        "seed": cfg.SEED,
        "mlflow_experiment": cfg.MLFLOW_EXPERIMENT,
        "run_name": cfg.RUN_NAME,
        "project": cfg.PROJECT_DIR,
        "lr0": cfg.LEARNING_RATE_INITIAL,
        "lrf": cfg.LEARNING_RATE_FINAL,
        "weight_decay": cfg.WEIGHT_DECAY,
        "warmup_epochs": cfg.WARMUP_EPOCHS,
        "patience": cfg.PATIENCE,
        "hsv_h": cfg.HSV_H,
        "hsv_s": cfg.HSV_S,
        "hsv_v": cfg.HSV_V,
        "degrees": cfg.DEGREES,
        "translate": cfg.TRANSLATE,
        "scale": cfg.SCALE,
        "shear": cfg.SHEAR,
        "perspective": cfg.PERSPECTIVE,
        "flipud": cfg.FLIPUD,
        "fliplr": cfg.FLIPLR,
        "mosaic": cfg.MOSAIC,
        "mixup": cfg.MIXUP,
        "copy_paste": cfg.COPY_PASTE,
        "amp": cfg.AMP,
        "cache": cfg.CACHE,
    }
    
    # Apply preset if specified
    if preset:
        preset_map = {
            "quick_test": cfg.PRESET_QUICK_TEST,
            "baseline": cfg.PRESET_BASELINE,
            "heavy_aug": cfg.PRESET_HEAVY_AUG,
        }
        
        if preset not in preset_map:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_map.keys())}")
        
        preset_config = preset_map[preset]
        params.update(preset_config)
        print(f"\n🎯 Using preset: {preset}")
    
    # Apply manual overrides
    if overrides:
        params.update(overrides)
        print(f"📝 Overrides: {list(overrides.keys())}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Dataset:     {params['dataset_yaml']}")
    print(f"Model:       {params['model_family']}{params['model_size']}")
    print(f"Epochs:      {params['epochs']}")
    print(f"Image size:  {params['imgsz']}")
    print(f"Batch:       {params['batch']}")
    print(f"Device:      {params['device']}")
    print(f"Experiment:  {params['mlflow_experiment']}")
    print(f"Run name:    {params['run_name']}")
    print("="*80 + "\n")
    
    # Run training
    return train_yolo(**params)