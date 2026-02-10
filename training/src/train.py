import tempfile
from pathlib import Path
from typing import Optional, Dict, List

import mlflow
from ultralytics import YOLO

# IMPORTANT: importe ta classe
from data_pipeline.src.validate_extract_data import DatasetValidatorExtractor


import zipfile
from PIL import Image


# ------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------

def _write_dataset_yaml(dataset_root: Path, class_names: List[str], yaml_path: Path) -> None:
    train_images = dataset_root / "train" / "images"
    val_images = dataset_root / "val" / "images"
    test_images = dataset_root / "test" / "images"

    if not train_images.exists():
        raise FileNotFoundError(f"Missing: {train_images}")
    if not val_images.exists():
        raise FileNotFoundError(f"Missing: {val_images}")

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
    if dir_path.exists():
        mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)


def _materialize_yolo_labels_from_zip(
    images_dir: Path,
    annotations_dir: Path,
    labels_out_dir: Path,
    strict: bool = True
) -> None:
    """
    Lit les annotations zip avec DatasetValidatorExtractor
    et écrit les labels .txt dans labels_out_dir, un par image.
    """
    extractor = DatasetValidatorExtractor(
        images_dir=str(images_dir),
        annotations_dir=str(annotations_dir),
    )

    ok = extractor.validate()
    if not ok and strict:
        raise ValueError(f"Dataset split invalid for {images_dir.parent.name}")

    images = extractor.get_images()
    annotations = extractor.get_annotations() or {}

    labels_out_dir.mkdir(parents=True, exist_ok=True)

    # écrit un .txt pour chaque image
    missing = 0
    for img_path in images:
        stem = img_path.stem
        txt = annotations.get(stem, None)

        if txt is None:
            missing += 1
            if strict:
                raise FileNotFoundError(
                    f"Missing annotation for image {img_path.name} in {annotations_dir}"
                )
            else:
                txt = ""  # fichier vide si non strict

        (labels_out_dir / f"{stem}.txt").write_text(txt, encoding="utf-8")

    if missing > 0:
        print(f"[WARN] {missing} missing annotations in {images_dir.parent.name}")


def _prepare_ultralytics_dataset_in_tmp(
    dataset_root: Path,
    tmp_root: Path,
    strict: bool = True
) -> Path:
    """
    Crée une structure YOLO standard dans tmp_root à partir de:
      dataset_root/train/images + train/annotations(zip)
      dataset_root/val/images   + val/annotations(zip)
      dataset_root/test/images  + test/annotations(zip)
    """
    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            if split == "test":
                continue
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        images_dir = split_dir / "images"
        annotations_dir = split_dir / "annotations"  # <= ton format zip

        if not images_dir.exists():
            raise FileNotFoundError(f"Missing: {images_dir}")
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Missing: {annotations_dir}")

        # Copie des images dans tmp (Ultralytics lit ici)
        out_images = tmp_root / split / "images"
        out_labels = tmp_root / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)

        for img in images_dir.iterdir():
            if img.is_file():
                # hardlink si possible (plus rapide), sinon copie
                try:
                    (out_images / img.name).hardlink_to(img)
                except Exception:
                    import shutil
                    shutil.copy2(img, out_images / img.name)

        # Création des labels .txt à partir du zip
        _materialize_yolo_labels_from_zip(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            labels_out_dir=out_labels,
            strict=strict
        )

    return tmp_root


# ------------------------------------------------------------
# Main train
# ------------------------------------------------------------

def train_yolo(
    dataset_root: str,
    class_names: List[str],
    model_family: str = "yolo26",     # "yolo11" ou "yolo26"
    model_size: str = "n",            # n/s/m/l/x
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    workers: int = 8,
    seed: int = 42,
    project: str = "runs",
    run_name: str = "exp",
    mlflow_experiment: str = "yolo_training",
    mlflow_run_name: Optional[str] = None,
    strict_validation: bool = True,

    # Hyperparams / augmentations
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

    amp: bool = True,
    cache: bool = False,
) -> None:

    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    model_ckpt = f"{model_family}{model_size}.pt"

    mlflow.set_experiment(mlflow_experiment)
    run_display_name = mlflow_run_name or f"{model_ckpt}_{run_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1) Matérialise dataset YOLO standard dans tmp
        tmp_dataset_root = tmpdir / "yolo_dataset"
        tmp_dataset_root.mkdir(parents=True, exist_ok=True)

        _prepare_ultralytics_dataset_in_tmp(
            dataset_root=dataset_root,
            tmp_root=tmp_dataset_root,
            strict=strict_validation
        )

        # 2) Crée dataset.yaml
        dataset_yaml = tmpdir / "dataset.yaml"
        _write_dataset_yaml(tmp_dataset_root, class_names, dataset_yaml)

        # 3) MLflow run
        with mlflow.start_run(run_name=run_display_name, log_system_metrics=True) as run:
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
                "strict_validation": strict_validation,
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

            # 4) Train
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

                lr0=lr0,
                lrf=lrf,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                patience=patience,

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

                amp=amp,
                cache=cache,
            )

            save_dir = Path(results.save_dir)

            # 5) Log artefacts Ultralytics
            _log_dir_as_artifacts(save_dir, artifact_path="ultralytics_run")

            best_pt = save_dir / "weights" / "best.pt"
            last_pt = save_dir / "weights" / "last.pt"

            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="weights")
            if last_pt.exists():
                mlflow.log_artifact(str(last_pt), artifact_path="weights")

            # 6) Eval sur test si présent
            if (tmp_dataset_root / "test" / "images").exists():
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
                except Exception:
                    pass

            # 7) Log requirements
            if Path("requirements.txt").exists():
                mlflow.log_artifact("requirements.txt", artifact_path="environment")

            print("\nTraining terminé.")
            print(f"MLflow run_id: {run.info.run_id}")
            print(f"Ultralytics save_dir: {save_dir}")


if __name__ == "__main__":
    train_yolo(
        dataset_root="output",
        class_names=["0", "1", "2", "3", "4", "5"],

        model_family="yolo26",
        model_size="n",
        epochs=80,
        imgsz=640,
        batch=16,
        device="0",
        workers=8,
        seed=42,

        mlflow_experiment="yolo_training",
        run_name="yolo26n_sharp",

        strict_validation=True
    )