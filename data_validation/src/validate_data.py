"""
Module de validation du dataset SHARP
Vérifie la cohérence, l'intégrité et les classes du dataset
"""
import zipfile
from pathlib import Path
from PIL import Image

class DatasetValidatorExtractor:
    """Class used to validate and extract a dataset"""

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
        self.images = []
        self.annotations = []

    def validate(self) -> bool:
        """
        Check if the dataset is validated. If not, print errors
        
        Returns:
            True if all is validated, else False
        """

        print("\nChecking files consistency")
        self._check_files_consistency()
        
        print("\nChecking files integrity")
        self._check_files_integrity()

        self._print_report()

        return len(self.errors) == 0

    def _check_files_consistency(self) -> None:
        """Check consistency between images and their annotations"""
        # Retrieve images
        self.images = self._retrieve_images()
        image_stems = {f.stem for f in self.images}

        # Retriever annotations
        self.annotations = self._retrieve_annotations()

        if self.annotations is None:
            self.errors.append("No annotations found")
            return

        # Checking consistency
        annotation_stems = set(self.annotations.keys())
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

    def _check_files_integrity(self) -> None:
        """Check image and annotation files integrity"""
        image_files = self._retrieve_images()
        corrupted_images = []
        invalid_dimensions = []

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    # Check if image can be loaded
                    img.verify()

                # Reopen to check dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width < 10 or height < 10:
                        invalid_dimensions.append(f"{img_path.name} ({width}x{height})")

            except Exception as e:
                corrupted_images.append(f"{img_path.name}: {str(e)}")

        if corrupted_images:
            self.errors.append(
                f"{len(corrupted_images)} corrupted image(s): {corrupted_images[:5]}..."
            )
        else:
            print("All images are valid")
        
        if invalid_dimensions:
            self.errors.append(
                f"{len(invalid_dimensions)} image(s) with invalid dimensions: "
                f"{invalid_dimensions[:5]}..."
            )
        
        # Check annotations integrity
        annotations = self._retrieve_annotations()
        if annotations:
            invalid_annotations = []

            for name, content in annotations.items():
                if not content or content.strip() == "":
                    invalid_annotations.append(f"{name}: empty file")

            if invalid_annotations:
                self.errors.append(
                    f"{len(invalid_annotations)} empty annotation(s): "
                    f"{invalid_annotations[:5]}..."
                )
            else:
                print("All annotations validated")

    def _retrieve_images(self):
        """Retrieve all image files"""
        image_files = []

        for ext in self.VALID_IMAGE_EXTENSIONS:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))

        return sorted(image_files)

    def _retrieve_annotations(self):
        """
        Retrieve annotations from a zip file
        
        Returns:
            Dictionnary {raw_file_name: annotation_content}
        """
        annotations = {}

        # Retrieving the zip file
        zip_files = list(self.annotations_dir.rglob("*.zip"))

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

    def get_images(self):
        """Return the images extracted"""
        return self.images

    def get_annotations(self):
        """Return the annotations extracted"""
        return self.annotations
