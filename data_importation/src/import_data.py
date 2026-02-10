from picsellia import Client
from dotenv import load_dotenv
import os

load_dotenv("config/.env")

PICSELLIA_TOKEN = os.getenv("PICSELLIA_TOKEN")
PICSELLIA_ORGANIZATION = os.getenv("PICSELLIA_ORGANIZATION")
PICSELLIA_DATASET = os.getenv("PICSELLIA_DATASET")
PICSELLIA_VERSION = os.getenv("PICSELLIA_VERSION")

client = Client(
    api_token=PICSELLIA_TOKEN,
    organization_name=PICSELLIA_ORGANIZATION
    )

dataset = client.get_dataset(name=PICSELLIA_DATASET)

dataset_version = dataset.get_version(PICSELLIA_VERSION)

output_dir = "./dataset"
dataset_version.download(
    target_path=output_dir,
    with_annotations=True
)

print(f"import_data -> Dataset downloaded to {output_dir}")