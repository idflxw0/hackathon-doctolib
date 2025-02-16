# Utilisation de l'image officielle Python
FROM python:3.10

# Définition du répertoire de travail
WORKDIR /app

# Copier le fichier de dépendances et installer les paquets
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers
COPY . .

# Exposer le port de l'API
EXPOSE 8000

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
