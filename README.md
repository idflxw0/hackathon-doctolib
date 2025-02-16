# 📌 Chatbot Doctolib - Priorisation des Urgences Médicales

## 🏥 Description du Projet

Ce projet est un **chatbot médical** intégré à Doctolib permettant aux patients d'expliquer leur **situation d'urgence** afin d'obtenir un rendez-vous plus rapidement avec :

- Leur **médecin traitant** ✅
- Un **autre médecin disponible** en cas d'indisponibilité ❌

Notre **IA détecte les personnes mentant sur leur état de santé** et les oriente uniquement vers un **consultant disponible**, garantissant ainsi un accès prioritaire aux véritables urgences.

---

## ⚙️ Technologies Utilisées

### Backend

- **FastAPI** - API REST rapide et asynchrone 🚀
- **Milvus** - Base de données vectorielle pour la recherche de symptômes 🧠
- **ETCD** - Stockage distribué des configurations ⚙️
- **MinIO** - Stockage d'objets pour les dossiers médicaux 📁

### Chatbot (NLP)

- **Python** (Flask pour le chatbot 🤖)
- **Transformers (Hugging Face)** - Modèles IA de compréhension du langage naturel 🏥
- **spaCy / NLTK** - Analyse de texte 📊

### DevOps & Conteneurisation

- **Docker & Docker-Compose** - Conteneurisation de l'API et du chatbot 🐳
- **Nginx** - Proxy et gestion des requêtes 🌐

### Frontend

- **React.js / Vue.js** (au choix) - Interface utilisateur intuitive 🎨
- **Fetch API / Axios** - Communication avec l'API 📨

---

## 🏗️ Architecture du Projet

```bash
.
├── backend
│   ├── app
│   │   ├── __init__.py
│   │   ├── main.py  # FastAPI pour gérer les requêtes
│   │   ├── chatbot.py  # Connexion avec le chatbot
│   │   ├── models.py  # Schéma Pydantic
│   ├── Dockerfile
│   ├── requirements.txt
├── chatbot
│   ├── app.py  # API Flask pour le chatbot
│   ├── model.pkl  # Modèle NLP entraîné
│   ├── Dockerfile
│   ├── requirements.txt
├── frontend
│   ├── src
│   │   ├── components
│   │   ├── pages
│   │   ├── App.js  # Interface principale
│   ├── package.json
│   ├── Dockerfile
├── docker-compose.yml  # Orchestration des services
├── README.md

```

---

## 🔧 Installation et Déploiement

### 1️⃣ Cloner le projet

```
git clone https://github.com/votre-repo/chatbot-doctolib.git
cd chatbot-doctolib

```

Warning :

Pour pouvoir utiliser la démo il faudra lanrcr les commandes suivant:

docker-compose up -d

npm run dev

Dans un autre terminal

$env : MISTRAL_API_KEY = "api_key"2️⃣ Lancer les conteneurs Docker

```
docker-compose up -d --build

```

✅ FastAPI est accessible sur [**http://localhost:8000**](http://localhost:8000/)
✅ Swagger API Docs : [**http://localhost:8000/docs**](http://localhost:8000/docs)
✅ Chatbot tourne sur [**http://localhost:5000**](http://localhost:5000/)

### 3️⃣ Tester l'API

Envoyer une requête au chatbot :

```
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message": "J’ai 40°C de fièvre et je tousse"}'

```

🔹 Réponse JSON :

```json
{
  "urgence": "élevée",
  "rdv": "Médecin traitant disponible sous 24h"
}

```

---

## 🛠️ Fonctionnalités du Chatbot

✔️ **Analyse des symptômes en langage naturel** 🤖
✔️ **Évaluation de l'urgence médicale** 🔥
✔️ **Priorisation des rendez-vous médicaux** 🏥
✔️ **Détection des tentatives de fraude** 🚫
✔️ **Interaction rapide et fluide** 💬

---

## 🔜 Améliorations Futures

- 🔹 Amélioration du modèle IA avec **BERT médical** 🏥
- 🔹 Intégration avec **Google Calendar** pour réservation automatique 📅
- 🔹 Ajout de la **reconnaissance vocale** 🎙️
