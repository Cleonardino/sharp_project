"""
YOLO Training Script with MLflow Integration
Trains YOLO models using a pre-prepared dataset with dataset.yaml
"""

import os
from pathlib import Path
from typing import Optional

import mlflow
from ultralytics import YOLO
from dotenv import load_dotenv


# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Training configuration"""
    
    # Dataset path
    DATASET_YAML = Path("prepared_data/dataset.yaml")
    
    # MLflow settings (loaded from .env)
    MLFLOW_TRACKING_URI: str = None
    
    @classmethod
    def load_env(cls, env_path: str = "config/.env"):
        """Load environment variables"""
        load_dotenv(env_path)
        cls.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        if cls.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(cls.MLFLOW_TRACKING_URI)


# ============================================================================
# Training Function
# ============================================================================

def train_yolo(
    # Dataset
    dataset_yaml: str = "prepared_data/dataset.yaml",
    
    # Model configuration
    model_family: str = "yolo11",
    model_size: str = "n",
    
    # Training parameters
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    workers: int = 8,
    seed: int = 42,
    
    # MLflow settings
    mlflow_experiment: str = "yolo_training",
    run_name: str = "exp",
    
    # Output directory
    project: str = "runs",
    
    # Optimizer hyperparameters
    lr0: float = 0.01,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    patience: int = 30,
    
    # Data augmentation
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 0,
    mixup: float = 0.1,
    copy_paste: float = 0.0,
    
    # Other settings
    amp: bool = True,
    cache: bool = False,
) -> str:
    """
    Train YOLO model with MLflow tracking
    
    Args:
        dataset_yaml: Path to dataset.yaml configuration file
        model_family: "yolo11" or "yolo26"
        model_size: "n", "s", "m", "l", or "x"
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device ("0" for GPU, "cpu", "mps" for Mac)
        workers: Number of data loading workers
        seed: Random seed for reproducibility
        mlflow_experiment: MLflow experiment name
        run_name: MLflow run name
        project: Ultralytics project directory
        (other parameters for optimizer and augmentation)
        
    Returns:
        MLflow run_id
    """
    
    # Verify dataset.yaml exists
    dataset_path = Path(dataset_yaml)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_path}")
    
    # Model checkpoint name
    model_checkpoint = f"{model_family}{model_size}.pt"
    
    # Setup MLflow
    mlflow.set_experiment(mlflow_experiment)
    run_display_name = f"{model_checkpoint}_{run_name}"
    
    print("\n" + "="*80)
    print("YOLO TRAINING WITH MLFLOW")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_checkpoint}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  MLflow experiment: {mlflow_experiment}")
    print(f"  Run name: {run_display_name}")
    print("="*80 + "\n")
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_display_name, log_system_metrics=True) as run:
        run_id = run.info.run_id
        
        # Log all parameters
        mlflow.log_params({
            "model": model_checkpoint,
            "model_family": model_family,
            "model_size": model_size,
            "dataset_yaml": str(dataset_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "seed": seed,
            "amp": amp,
            "cache": cache,
        })
        
        mlflow.log_params({
            "lr0": lr0,
            "lrf": lrf,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "patience": patience,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "shear": shear,
            "perspective": perspective,
            "flipud": flipud,
            "fliplr": fliplr,
            "mosaic": mosaic,
            "mixup": mixup,
            "copy_paste": copy_paste,
        })
        
        print("Training started...")
        print("-"*80 + "\n")
        
        # Initialize and train model
        model = YOLO(model_checkpoint)
        
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            seed=seed,
            project=project,
            name=run_name,
            exist_ok=True,
            
            # Optimizer
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            patience=patience,
            
            # Data augmentation
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
            
            # Other
            amp=amp,
            cache=cache,
            )
        
        save_dir = Path(results.save_dir)
        
        print("\n" + "-"*80)
        print("TRAINING COMPLETED")
        print("-"*80)
        print(f"Results saved to: {save_dir}")
        
        # Log Ultralytics artifacts
        print("\nLogging artifacts to MLflow...")
        if save_dir.exists():
            mlflow.log_artifacts(str(save_dir), artifact_path="ultralytics_run")
            print(f"  ✓ Logged all training artifacts")
        
        # Log model weights separately for easy access
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"
        
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            print(f"  ✓ Logged best.pt")
        if last_pt.exists():
            mlflow.log_artifact(str(last_pt), artifact_path="weights")
            print(f"  ✓ Logged last.pt")
        
        # Evaluate on test set if available
        dataset_root = dataset_path.parent
        test_images_dir = dataset_root / "test" / "images"
        
        if test_images_dir.exists():
            print("\nEvaluating on test set...")
            try:
                test_metrics = model.val(
                    data=str(dataset_yaml),
                    split="test",
                    imgsz=imgsz,
                    device=device
                )
                
                mlflow.log_metric("test_map50_95", float(test_metrics.box.map))
                mlflow.log_metric("test_map50", float(test_metrics.box.map50))
                mlflow.log_metric("test_map75", float(test_metrics.box.map75))
                print(f"Test mAP50-95: {test_metrics.box.map:.4f}")
                print(f"Test mAP50:    {test_metrics.box.map50:.4f}")
            except Exception as e:
                print(f"  ⚠ Could not evaluate on test set: {e}")
        
        # Log requirements if available
        req_file = Path("requirements.txt")
        if req_file.exists():
            mlflow.log_artifact(str(req_file), artifact_path="environment")
            print(f"  ✓ Logged requirements.txt")
        
        print(f"\n" + "="*80)
        print("MLflow Run Completed")
        print("="*80)
        print(f"Run ID: {run_id}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print("="*80 + "\n")
        
        return run_id