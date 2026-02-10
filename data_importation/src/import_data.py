"""
This module proposes a function to download the dataset from picsellia, dowload_dataset
"""
import os
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from dotenv import load_dotenv
import config.config as cf

def download_dataset():
    """
    Download the picsellia dataset using the key in the .env file
    and the values in the config.py file.
    """

    load_dotenv("config/.env")

    picsellia_token = os.getenv("PICSELLIA_TOKEN")

    client = Client(
        api_token=picsellia_token,
        organization_name=cf.PICSELLIA_ORGANIZATION
        )

    dataset = client.get_dataset(name=cf.PICSELLIA_DATASET)

    dataset_version = dataset.get_version(cf.PICSELLIA_VERSION)

    image_dir = "./dataset/images"
    annotations_dir = "./dataset/annotations"
    dataset_version.download(
        target_path=image_dir
    )

    print(f"import_data -> Dataset downloaded to {image_dir}")

    dataset_version.export_annotation_file(
        annotation_file_type=AnnotationFileType.YOLO,
        target_path=annotations_dir
    )

    print(f"import_data -> YOLO annotations exported to {annotations_dir}")
