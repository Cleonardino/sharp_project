import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
from ultralytics import YOLO


def _write_dataset_yaml(
    dataset_root: Path,
    class_names: list[str],
    yaml_path: Path,
) -> None:
    """
    Crée un dataset.yaml compatible Ultralytics.

    Structure attendue:
    dataset_root/
        train/images
        train/labels
        val/images
        val/labels
        test/images
        test/labels
    """
    train_images = dataset_root / "train" / "images"
    val_images = dataset_root / "val" / "images"
    test_images = dataset_root / "test" / "images"

    if not train_images.exists():
        raise FileNotFoundError(f"Missing: {train_images}")
    if not val_images.exists():
        raise FileNotFoundError(f"Missing: {val_images}")
    if not test_images.exists():
        print("[WARN] No test split found. Ultralytics will train without test metrics.")

    # Ultralytics YAML minimal
    lines = []
    lines.append(f"path: {dataset_root.as_posix()}")
    lines.append("train: train/images")
    lines.append("val: val/images")
    if test_images.exists():
        lines.append("test: test/images")

    lines.append(f"nc: {len(class_names)}")
    lines.append("names:")
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    yaml_path.write_text("\n".join(lines), encoding="utf-8")


def _log_dir_as_artifacts(dir_path: Path, artifact_path: str) -> None:
    """Log récursivement un dossier dans MLflow."""
    if not dir_path.exists():
        return
    mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)


def train_yolo(
    dataset_root: str,
    class_names: list[str],
    model_family: str = "yolo11",         # "yolo11" ou "yolo26"
    model_size: str = "n",                # n/s/m/l/x
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",                    # "0", "cpu", "mps"
    workers: int = 8,
    seed: int = 42,
    project: str = "runs",
    run_name: str = "exp",
    mlflow_experiment: str = "yolo_training",
    mlflow_run_name: Optional[str] = None,

    # Hyperparams / augmentations (tu peux les tuner)
    lr0: float = 0.01,
    lrf: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    patience: int = 30,

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

    # perf
    amp: bool = True,
    cache: bool = False,   # True si dataset pas énorme et tu veux accélérer
) -> None:
    """
    Entraîne YOLO (Ultralytics) sur un dataset déjà split.
    Log complet MLflow.
    """

    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # ---- 1) Crée dataset.yaml (dans un dossier temporaire pour éviter de polluer)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dataset_yaml = tmpdir / "dataset.yaml"
        _write_dataset_yaml(dataset_root, class_names, dataset_yaml)

        # ---- 2) Choix du modèle
        # Ex: yolo11n.pt, yolo26s.pt ...
        model_ckpt = f"{model_family}{model_size}.pt"

        # ---- 3) MLflow
        mlflow.set_experiment(mlflow_experiment)

        run_display_name = mlflow_run_name or f"{model_ckpt}_{run_name}"

        with mlflow.start_run(run_name=run_display_name, log_system_metrics=True) as run:
            # Log paramètres principaux
            mlflow.log_params({
                "model": model_ckpt,
                "dataset_root": str(dataset_root),
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
                "workers": workers,
                "seed": seed,
                "amp": amp,
                "cache": cache,
            })

            # Log hyperparams / augmentations
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

            # ---- 4) Train
            model = YOLO(model_ckpt)

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

                # optimisation
                lr0=lr0,
                lrf=lrf,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                patience=patience,

                # augmentations
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

                # perf
                amp=amp,
                cache=cache,)

            # ---- 5) Ultralytics écrit les résultats dans runs/detect/...
            # On récupère le dossier du run
            save_dir = Path(results.save_dir)

            # ---- 6) Log des artefacts importants
            _log_dir_as_artifacts(save_dir, artifact_path="ultralytics_run")

            # Log best.pt / last.pt plus explicitement
            best_pt = save_dir / "weights" / "best.pt"
            last_pt = save_dir / "weights" / "last.pt"

            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="weights")
            if last_pt.exists():
                mlflow.log_artifact(str(last_pt), artifact_path="weights")

            # ---- 7) Evaluation sur test (si présent)
            test_images = dataset_root / "test" / "images"
            if test_images.exists():
                test_metrics = model.val(
                    data=str(dataset_yaml),
                    split="test",
                    imgsz=imgsz,
                    device=device
                )

                # Log quelques métriques test (les noms varient selon versions Ultralytics)
                # On log ce qu'on peut récupérer sans casser le script.
                try:
                    # mAP50-95, mAP50 etc.
                    mlflow.log_metric("test_map50_95", float(test_metrics.box.map))
                    mlflow.log_metric("test_map50", float(test_metrics.box.map50))
                    mlflow.log_metric("test_map75", float(test_metrics.box.map75))
                except Exception:
                    pass

            # ---- 8) Log requirements (comme ton exemple)
            if Path("requirements.txt").exists():
                mlflow.log_artifact("requirements.txt", artifact_path="environment")

            print("\nTraining terminé.")
            print(f"MLflow run_id: {run.info.run_id}")
            print(f"Ultralytics save_dir: {save_dir}")


if __name__ == "__main__":
    # Exemple : adapte class_names à ton dataset
    train_yolo(
        dataset_root="output",  # ex: output/train/images ...
        class_names=["class0", "class1"],

        model_family="yolo26",
        model_size="n",  # nano = plus rapide => plus facile d'atteindre 10 FPS
        epochs=80,
        imgsz=640,
        batch=16,
        device="0",      # GPU NVIDIA: "0" / Mac: "mps" / CPU: "cpu"
        workers=8,
        seed=42,

        mlflow_experiment="yolo_training",
        run_name="yolo26n_custom"
    )