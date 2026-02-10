import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from sklearn.model_selection import train_test_split


@dataclass
class SampleInRAM:
    """Un sample en RAM : image + annotation YOLO"""
    stem: str
    image_bytes: bytes
    annotation_txt: str


class DatasetSplitter:
    """Class used to split a dataset into train/val/test sets"""

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        output_dir: str,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        class_names: List[str] = ["0", "1", "2", "3", "4", "5"],
        seed: int = 42,
        skip_missing_annotations: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.class_names = class_names
        self.seed = seed
        self.skip_missing_annotations = skip_missing_annotations

        self.errors: List[str] = []
        self._annotations_cache: Optional[Dict[str, str]] = None

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations dir not found: {self.annotations_dir}")

    # -------------------------
    # Helpers
    # -------------------------
    def _find_images(self) -> List[Path]:
        """Trouve toutes les images (png/jpg/jpeg)"""
        exts = ["*.png", "*.jpg", "*.jpeg"]
        image_files: List[Path] = []
        for ext in exts:
            image_files.extend(self.images_dir.glob(ext))
        return sorted(image_files)

    def _get_annotations(self) -> Optional[Dict[str, str]]:
        """
        Retrieve annotations from .txt files directly in the annotations directory

        Returns:
            Dict {stem: annotation_content}
        """
        annotations: Dict[str, str] = {}

        # Lire directement les fichiers .txt dans le répertoire annotations
        txt_files = sorted(self.annotations_dir.glob("*.txt"))

        if not txt_files:
            self.errors.append("No .txt annotation files found in annotations directory")
            return None

        print(f"Reading {len(txt_files)} annotation files from {self.annotations_dir}")

        for txt_file in txt_files:
            # Ignorer data.yaml ou autres fichiers non-annotations
            if txt_file.name == "data.yaml" or txt_file.name.startswith("_"):
                continue
            
            stem = txt_file.stem
            try:
                content = txt_file.read_text(encoding="utf-8")
                annotations[stem] = content
            except Exception as e:
                self.errors.append(f"Error reading {txt_file.name}: {e}")

        if len(annotations) == 0:
            self.errors.append("No valid .txt annotations found")
            return None

        print(f"Loaded {len(annotations)} annotations")
        return annotations

    def _ensure_annotations_loaded(self) -> None:
        """Charge les annotations une seule fois (cache en RAM)."""
        if self._annotations_cache is None:
            self._annotations_cache = self._get_annotations()

    def _load_sample_in_ram(self, image_path: Path) -> Optional[SampleInRAM]:
        """
        Charge une image + annotation YOLO correspondante en RAM.
        Annotation récupérée depuis les fichiers .txt
        """
        self._ensure_annotations_loaded()

        if self._annotations_cache is None:
            # Pas d'annotations trouvées
            return None

        stem = image_path.stem

        if stem not in self._annotations_cache:
            if self.skip_missing_annotations:
                return None
            raise FileNotFoundError(
                f"Missing annotation .txt for image {stem} in {self.annotations_dir}"
            )

        image_bytes = image_path.read_bytes()
        annotation_txt = self._annotations_cache[stem]

        return SampleInRAM(
            stem=stem,
            image_bytes=image_bytes,
            annotation_txt=annotation_txt
        )

    # -------------------------
    # Main
    # -------------------------
    def split(self) -> Tuple[List[SampleInRAM], List[SampleInRAM], List[SampleInRAM]]:
        """
        Split dataset and return 3 datasets fully in RAM:
        train, val, test
        """
        image_files = self._find_images()

        if len(image_files) == 0:
            raise ValueError(f"No images found in: {self.images_dir}")

        # 1) split train vs (val+test)
        train_files, temp_files = train_test_split(
            image_files,
            train_size=self.train_ratio,
            random_state=self.seed,
            shuffle=True
        )

        # 2) split temp into val and test
        temp_ratio = self.val_ratio + self.test_ratio
        val_size_in_temp = self.val_ratio / temp_ratio

        val_files, test_files = train_test_split(
            temp_files,
            train_size=val_size_in_temp,
            random_state=self.seed,
            shuffle=True
        )

        def load_split(files: List[Path]) -> List[SampleInRAM]:
            split_data: List[SampleInRAM] = []
            for img_path in files:
                sample = self._load_sample_in_ram(img_path)
                if sample is not None:
                    split_data.append(sample)
            return split_data

        train = load_split(train_files)
        val = load_split(val_files)
        test = load_split(test_files)

        return train, val, test

    def write_splits_to_disk(self,
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
        
        train_data, val_data, test_data = self.split()

        def write_split(samples: List[SampleInRAM], split_name: str) -> None:
            """Write a single split to disk"""
            images_dir = self.output_dir / split_name / "images"
            labels_dir = self.output_dir / split_name / "labels"
            annotations_dir = self.output_dir / split_name / "annotations"
            
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
        print(f"✓ All splits written to: {self.output_dir}")
        
        self._write_dataset_yaml()

        return self.output_dir

    def _write_dataset_yaml(self) -> None:
        """Write YOLO dataset.yaml configuration file"""
        train_images = self.output_dir / "train" / "images"
        val_images = self.output_dir / "val" / "images"
        test_images = self.output_dir / "test" / "images"
        
        if not train_images.exists():
            raise FileNotFoundError(f"Missing: {train_images}")
        if not val_images.exists():
            raise FileNotFoundError(f"Missing: {val_images}")
        
        lines = [
            f"path: {self.output_dir.as_posix()}",
            "train: train/images",
            "val: val/images",
        ]
        
        if test_images.exists():
            lines.append("test: test/images")
        
        lines.extend([
            f"nc: {len(self.class_names)}",
            "names:",
        ])
        
        for i, name in enumerate(self.class_names):
            lines.append(f"  {i}: {name}")
        yaml_path = self.output_dir / "dataset.yaml"
        yaml_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"✓ Dataset YAML created: {yaml_path}")

if __name__ == "__main__":
    splitter = DatasetSplitter(
        images_dir="dataset/images",
        annotations_dir="dataset/annotations",
        output_dir="output",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        skip_missing_annotations=False
    )

    train_data, val_data, test_data = splitter.split()

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    if splitter.errors:
        print("\nErrors:")
        for e in splitter.errors:
            print(" -", e)