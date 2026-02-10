from picsellia import Client
from dotenv import load_dotenv
import os

load_dotenv("config/.env")

PICSELLIA_TOKEN = os.getenv("PICSELLIA_TOKEN")
ORGANIZATION = os.getenv("ORGANIZATION")
DATASET = os.getenv("DATASET")

client = Client(
    api_token=PICSELLIA_TOKEN,
    organization_name=ORGANIZATION
    )

dataset = client.get_dataset(name=DATASET)