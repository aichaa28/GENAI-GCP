# Utiliser une image de base officielle Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt --timeout=500 --verbose

# Copier tous les fichiers de ton projet dans le conteneur
COPY . .

EXPOSE 8000
# Définir les variables d'environnement à partir du fichier .env

RUN pip install python-dotenv

# Exposer le port si nécessaire


# Commande par défaut pour exécuter ton application
CMD ["streamlit", "run", "app.py"]
