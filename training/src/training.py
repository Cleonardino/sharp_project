"""
Complete YOLO Training Pipeline with MLflow Integration
Orchestrates: Data Import -> Preparation -> Validation -> Training -> MLflow Logging
"""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional
import sys
import mlflow
from ultralytics import YOLO
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import custom modules (assumed to exist in data_pipeline package)
from data_pipeline.src.import_data import download_dataset
from data_pipeline.src.prepare_data import DatasetSplitter, SampleInRAM
from data_pipeline.src.validate_extract_data import DatasetValidatorExtractor


# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Central configuration for the training pipeline"""
    
    # Picsellia Dataset Settings (loaded from .env)
    PICSELLIA_TOKEN: str = None
    PICSELLIA_ORGANIZATION: str = "clement4"
    PICSELLIA_DATASET: str = "Datasets/HandCounting"
    PICSELLIA_VERSION: str = "v1"
    
    # Paths
    RAW_DATA_DIR = Path("dataset")
    RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
    RAW_ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"
    OUTPUT_DIR = Path("output")
    
    # Split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    SEED = 42
    
    # Classes
    CLASS_NAMES = ["0", "1", "2", "3", "4", "5"]
    
    @classmethod
    def load_env(cls, env_path: str = "config/.env"):
        """Load environment variables"""
        load_dotenv(env_path)
        cls.PICSELLIA_TOKEN = os.getenv("PICSELLIA_TOKEN")
        if not cls.PICSELLIA_TOKEN:
            raise ValueError("PICSELLIA_TOKEN not found in .env file")


# ============================================================================
# Data Pipeline Functions
# ============================================================================

def step_1_import_data(config: TrainingConfig, force_download: bool = False) -> None:
    """
    Step 1: Download dataset from Picsellia
    
    Args:
        config: Training configuration
        force_download: If True, re-download even if data exists
    """
    print("\n" + "="*80)
    print("STEP 1: IMPORTING DATA FROM PICSELLIA")
    print("="*80)
    
    if config.RAW_IMAGES_DIR.exists() and not force_download:
        print(f"✓ Data already exists at {config.RAW_IMAGES_DIR}")
        print("  Use force_download=True to re-download")
        return
    
    # Create directories
    config.RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download from Picsellia
    download_dataset(
        output_image_dir=str(config.RAW_IMAGES_DIR),
        output_annotations_dir=str(config.RAW_ANNOTATIONS_DIR)
    )
    
    print(f"✓ Data imported successfully")
    print(f"  Images: {config.RAW_IMAGES_DIR}")
    print(f"  Annotations: {config.RAW_ANNOTATIONS_DIR}")


def step_2_split_data(config: TrainingConfig) -> tuple:
    """
    Step 2: Split dataset into train/val/test
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    print("\n" + "="*80)
    print("STEP 2: SPLITTING DATA INTO TRAIN/VAL/TEST")
    print("="*80)
    
    splitter = DatasetSplitter(
        images_dir=str(config.RAW_IMAGES_DIR),
        annotations_dir=str(config.RAW_ANNOTATIONS_DIR),
        output_dir=str(config.OUTPUT_DIR),
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SEED,
        skip_missing_annotations=False
    )
    
    train_data, val_data, test_data = splitter.split()
    
    print(f"✓ Dataset split completed:")
    print(f"  Train: {len(train_data)} samples ({config.TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {len(val_data)} samples ({config.VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {len(test_data)} samples ({config.TEST_RATIO*100:.0f}%)")
    
    if splitter.errors:
        print("\n⚠ Warnings during split:")
        for error in splitter.errors:
            print(f"  - {error}")
    
    return train_data, val_data, test_data


def step_3_write_splits_to_disk(
    train_data: List[SampleInRAM],
    val_data: List[SampleInRAM],
    test_data: List[SampleInRAM],
    output_dir: Path
) -> Path:
    """
    Step 3: Write split data to disk in YOLO format
    
    Args:
        train_data: Training samples in RAM
        val_data: Validation samples in RAM
        test_data: Test samples in RAM
        output_dir: Root directory for output
        
    Returns:
        Path to the dataset root
    """
    print("\n" + "="*80)
    print("STEP 3: WRITING SPLITS TO DISK")
    print("="*80)
    
    def write_split(samples: List[SampleInRAM], split_name: str) -> None:
        """Write a single split to disk"""
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        annotations_dir = output_dir / split_name / "annotations"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Write images, labels and annotations
        for sample in samples:
            # Write image
            image_path = images_dir / f"{sample.stem}.jpg"
            image_path.write_bytes(sample.image_bytes)
            
            # Write label (for YOLO training)
            label_path = labels_dir / f"{sample.stem}.txt"
            label_path.write_text(sample.annotation_txt, encoding="utf-8")
            
            # Write annotation (for validation compatibility)
            annotation_path = annotations_dir / f"{sample.stem}.txt"
            annotation_path.write_text(sample.annotation_txt, encoding="utf-8")
        
        print(f"  ✓ {split_name:5s}: {len(samples):4d} samples written")
    
    write_split(train_data, "train")
    write_split(val_data, "val")
    write_split(test_data, "test")
    
    print(f"✓ All splits written to: {output_dir}")
    return output_dir


# ============================================================================
# Training Functions (adapted from train.py)
# ============================================================================

def _write_dataset_yaml(dataset_root: Path, class_names: List[str], yaml_path: Path) -> None:
    """Write YOLO dataset.yaml configuration file"""
    train_images = dataset_root / "train" / "images"
    val_images = dataset_root / "val" / "images"
    test_images = dataset_root / "test" / "images"
    
    if not train_images.exists():
        raise FileNotFoundError(f"Missing: {train_images}")
    if not val_images.exists():
        raise FileNotFoundError(f"Missing: {val_images}")
    
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: train/images",
        "val: val/images",
    ]
    
    if test_images.exists():
        lines.append("test: test/images")
    
    lines.extend([
        f"nc: {len(class_names)}",
        "names:",
    ])
    
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Dataset YAML created: {yaml_path}")


def _log_dir_as_artifacts(dir_path: Path, artifact_path: str) -> None:
    """Log directory contents as MLflow artifacts"""
    if dir_path.exists():
        mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)


def step_5_train_yolo(
    dataset_root: Path,
    class_names: List[str],
    
    # Model configuration
    model_family: str = "yolo11",
    model_size: str = "n",
    
    # Training parameters
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    workers: int = 8,
    seed: int = 42,
    
    # MLflow settings
    mlflow_experiment: str = "yolo_training",
    run_name: str = "exp",
    
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
    mosaic: float = 1.0,
    mixup: float = 0.1,
    copy_paste: float = 0.0,
    
    # Other settings
    amp: bool = True,
    cache: bool = False,
    project: str = "runs",
    strict_validation: bool = True,
) -> str:
    """
    Step 5: Train YOLO model with MLflow tracking
    
    Returns:
        MLflow run_id
    """
    print("\n" + "="*80)
    print("STEP 5: TRAINING YOLO MODEL")
    print("="*80)
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    model_checkpoint = f"{model_family}{model_size}.pt"
    
    # Setup MLflow
    mlflow.set_experiment(mlflow_experiment)
    run_display_name = f"{model_checkpoint}_{run_name}"
    
    print(f"\nTraining Configuration:")
    print(f"  Model: {model_checkpoint}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  MLflow experiment: {mlflow_experiment}")
    print(f"  Run name: {run_display_name}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dataset.yaml
        dataset_yaml = tmpdir / "dataset.yaml"
        _write_dataset_yaml(dataset_root, class_names, dataset_yaml)
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_display_name, log_system_metrics=True) as run:
            run_id = run.info.run_id
            
            # Log all parameters
            mlflow.log_params({
                "model": model_checkpoint,
                "model_family": model_family,
                "model_size": model_size,
                "dataset_root": str(dataset_root),
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
                "workers": workers,
                "seed": seed,
                "amp": amp,
                "cache": cache,
                "strict_validation": strict_validation,
                "num_classes": len(class_names),
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
            
            print("\n" + "-"*80)
            print("TRAINING STARTED")
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
            _log_dir_as_artifacts(save_dir, artifact_path="ultralytics_run")
            
            # Log model weights
            best_pt = save_dir / "weights" / "best.pt"
            last_pt = save_dir / "weights" / "last.pt"
            
            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="weights")
                print(f"  ✓ Logged best.pt")
            if last_pt.exists():
                mlflow.log_artifact(str(last_pt), artifact_path="weights")
                print(f"  ✓ Logged last.pt")
            
            # Evaluate on test set if available
            test_images_dir = dataset_root / "test" / "images"
            if test_images_dir.exists():
                print("\nEvaluating on test set...")
                test_metrics = model.val(
                    data=str(dataset_yaml),
                    split="test",
                    imgsz=imgsz,
                    device=device
                )
                
                try:
                    mlflow.log_metric("test_map50_95", float(test_metrics.box.map))
                    mlflow.log_metric("test_map50", float(test_metrics.box.map50))
                    mlflow.log_metric("test_map75", float(test_metrics.box.map75))
                    print(f"  ✓ Test mAP50-95: {test_metrics.box.map:.4f}")
                    print(f"  ✓ Test mAP50:    {test_metrics.box.map50:.4f}")
                except Exception as e:
                    print(f"  ⚠ Could not log test metrics: {e}")
            
            # Log requirements if available
            req_file = Path("requirements.txt")
            if req_file.exists():
                mlflow.log_artifact(str(req_file), artifact_path="environment")
                print(f"  ✓ Logged requirements.txt")
            
            print(f"\n✓ MLflow run completed: {run_id}")
            print(f"  View at: {mlflow.get_tracking_uri()}")
            
            return run_id


# ============================================================================
# Main Pipeline
# ============================================================================

def run_complete_pipeline(
    # Data settings
    force_download: bool = False,
    skip_validation: bool = False,
    
    # Model settings
    model_family: str = "yolo11",
    model_size: str = "n",
    
    # Training settings
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    
    # MLflow settings
    mlflow_experiment: str = "yolo_training",
    run_name: str = "sharp_exp",
    
    # Augmentation settings
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.1,
    
    # Optimizer settings
    lr0: float = 0.01,
    patience: int = 30,
) -> str:
    """
    Run complete training pipeline from data import to model training
    
    Args:
        force_download: Re-download data even if exists
        skip_validation: Skip validation step (not recommended)
        model_family: "yolo11" or "yolo26"
        model_size: "n", "s", "m", "l", or "x"
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device ("0", "cpu", "mps", etc.)
        mlflow_experiment: MLflow experiment name
        run_name: MLflow run name
        (other augmentation and optimizer parameters)
        
    Returns:
        MLflow run_id
    """
    print("\n" + "="*80)
    print("YOLO TRAINING PIPELINE - STARTING")
    print("="*80)
    
    # Initialize configuration
    config = TrainingConfig()
    config.load_env()
    
    # Step 1: Import data
    step_1_import_data(config, force_download=force_download)
    
    # Step 2: Split data
    train_data, val_data, test_data = step_2_split_data(config)
    
    # Step 3: Write to disk
    dataset_root = step_3_write_splits_to_disk(
        train_data, val_data, test_data, config.OUTPUT_DIR
    )
    
    # Step 5: Train
    run_id = step_5_train_yolo(
        dataset_root=dataset_root,
        class_names=config.CLASS_NAMES,
        model_family=model_family,
        model_size=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        mlflow_experiment=mlflow_experiment,
        run_name=run_name,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        lr0=lr0,
        patience=patience,
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"MLflow Run ID: {run_id}")
    print(f"Dataset saved at: {dataset_root}")
    print("="*80 + "\n")
    
    return run_id


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example 1: Run complete pipeline with YOLO11
    run_id = run_complete_pipeline(
        force_download=False,  # Set to True to re-download data
        skip_validation=False,
        
        # Model
        model_family="yolo11",
        model_size="n",
        
        # Training
        epochs=80,
        imgsz=640,
        batch=16,
        device="0",  # Use "mps" for Mac, "cpu" for CPU
        
        # MLflow
        mlflow_experiment="sharp_yolo_training",
        run_name="yolo11n_baseline",
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        # Optimizer
        lr0=0.01,
        patience=30,
    )
    
    print(f"\n✓ Training completed! Run ID: {run_id}")
    
    # Example 2: Run with YOLO26 (uncomment to use)
    # run_id = run_complete_pipeline(
    #     model_family="yolo26",
    #     model_size="s",
    #     epochs=100,
    #     mlflow_experiment="sharp_yolo_training",
    #     run_name="yolo26s_experiment",
    #     device="mps",  # For Mac
    # )