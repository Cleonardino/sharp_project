from picsellia import Client
from dotenv import load_dotenv
import os
import data_importation.src.config as cf

def download_dataset():
    load_dotenv("config/.env")

    PICSELLIA_TOKEN = os.getenv("PICSELLIA_TOKEN")

    client = Client(
        api_token=PICSELLIA_TOKEN,
        organization_name=cf.PICSELLIA_ORGANIZATION
        )

    dataset = client.get_dataset(name=cf.PICSELLIA_DATASET)

    dataset_version = dataset.get_version(cf.PICSELLIA_VERSION)

    output_dir = "./dataset"
    dataset_version.download(
        target_path=output_dir
    )

    print(f"import_data -> Dataset downloaded to {output_dir}")