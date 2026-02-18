FROM python:3.12-slim

WORKDIR /app

# Copier le code training dans le conteneur
COPY ./src /app/src

# Installer les dépendances
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir mlflow pymysql boto3 torch torchvision

# Installer curl pour le script wait-for-it
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Ajouter le script wait-for-it
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Définir le point d'entrée
WORKDIR /app/src
CMD ["/wait-for-it.sh", "mysql-db:3306", "--", "/wait-for-it.sh", "minio:9000", "--", "python", "run_train.py"]
