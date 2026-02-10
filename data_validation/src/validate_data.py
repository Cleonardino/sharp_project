"""
Module de validation du dataset SHARP
Vérifie la cohérence, l'intégrité et les classes du dataset
"""
import zipfile
from pathlib import Path

class DatasetValidator:
    """Class used to validate a dataset"""

    VALID_CLASSES = {
        "0",
        "1", 
        "2",
        "3",
        "4",
        "5"
    }

    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, images_dir: str, annotations_dir: str):
        """
		Class init
        
        Args:
            images_dir: Path to directory containing images
            annotations_dir: Path to directory containing annotations
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.errors = []

    def validate(self) -> bool:
        """
        Check if the dataset is validated. If not, print errors
        
        Returns:
            True if all is validated, else False
        """

        print("\nChecking files consistency")
        self._check_files_consistency()

        self._print_report()

        return len(self.errors) == 0

    def _check_files_consistency(self) -> None:
        """Check consistency between images and their annotations"""
        # Retrieve images
        image_files = self._get_image_files()
        image_stems = {f.stem for f in image_files}

        # Retriever annotations
        annotations = self._get_annotations()

        if annotations is None:
            self.errors.append("No annotations found")
            return

        # Checking consistency
        annotation_stems = set(annotations.keys())
        # Images without annotations
        images_without_annotations = image_stems - annotation_stems
        if images_without_annotations:
            self.errors.append(
                f"{len(images_without_annotations)} image(s) without annotation.",
                "Here some examples:",
                f"{list(images_without_annotations)[:5]}..."
            )

        # Annotations without images
        annotations_without_images = annotation_stems - image_stems
        if annotations_without_images:
            self.errors.append(
                f"{len(annotations_without_images)} annotation(s) without image: "
                f"{list(annotations_without_images)[:5]}..."
            )

        print(f"{len(annotation_stems)} annotations found")
        print(f"{len(image_stems & annotation_stems)} image/annotation couples validated")

    def _get_image_files(self):
        """Get all image files"""
        image_files = []

        for ext in self.VALID_IMAGE_EXTENSIONS:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))

        return sorted(image_files)

    def _get_annotations(self):
        """
        Retrieve annotations from a zip file
        
        Returns:
            Dictionnary {raw_file_name: annotation_content}
        """
        annotations = {}

        # Retrieving the zip file
        zip_files = list(self.annotations_dir.glob("*.zip"))

        if zip_files:
            zip_path = zip_files[0]
            print(f"Reading annotations from {zip_path.name}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith('.txt'):
                        stem = Path(file_info.filename).stem
                        content = zip_ref.read(file_info.filename).decode('utf-8')
                        annotations[stem] = content

        else:
            self.errors.append("No zip annotation file found")
            return None

        return annotations

    def _print_report(self) -> None:
        """Print the report"""
        print("\n" + "=" * 60)
        print("=" * 60)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")

        print("\n" + "=" * 60)
        if len(self.errors) == 0:
            print("Validation successfull")
        else:
            print("Validation failed")
        print("=" * 60 + "\n")
