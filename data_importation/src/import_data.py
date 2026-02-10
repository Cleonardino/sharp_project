from picsellia import Client
from dotenv import load_dotenv
import os

load_dotenv("config/.env")

PICSELLIA_TOKEN = os.getenv("PICSELLIA_TOKEN")
ORGANIZATION = "clement4"

client = Client(
    api_token=PICSELLIA_TOKEN,
    organization_name=ORGANIZATION
    )
