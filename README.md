# 🚀 Système de Détection de Spam en Français

## 📍 Institut

**Nom**: ISPM – Institut Supérieur Polytechnique de Madagascar  
**Site Web**: [www.ispm-edu.com](https://www.ispm-edu.com)

---

## 👥 Équipe du Projet

| Nom | Rôle |
|-----|------|
| [Tianarimbola Andraina] | Chef de Projet / Responsable ML |
| [RAZANAPARANY Toky Faniriantsoa] | Développeur Backend (Flask) |
| [RANOMENJANAHARY Frederic Emmanuel] | Développeur Frontend (Web) |




---

## 🛠️ Stack Technologique

### Backend & ML
- **Python 3.x** - Langage de programmation principal
- **Flask 2.3.0** - Framework web pour l'API de prédiction
- **Flask-CORS 4.0.0** - Gestion des requêtes cross-origin
- **scikit-learn 1.3.0** - Bibliothèque ML pour l'entraînement et la prédiction
- **Joblib 1.3.0** - Sérialisation du modèle et vectorizer
- **Gunicorn 21.2.0** - Serveur de production WSGI

### Frontend
- **HTML5** - Structure des pages web
- **CSS3** - Styling et responsive design
- **JavaScript (Vanilla)** - Interactivité côté client

### DevOps / Deployment
- **Render** - Plateforme de déploiement (configuration: render.yaml, render-build.sh)
- **Package.json** - Gestion des scripts et dépendances Node.js

---

## 📊 Description du Processus et du Modèle

### Flux de Traitement

```
1. Collecte des Données
   ↓
2. Nettoyage et Filtrage (French Language)
   ↓
3. Prétraitement du Texte
   ├─ Tokenisation
   ├─ Suppression des stop-words français
   └─ Normalisation
   ↓
4. Vectorisation TF-IDF
   ↓
5. Entraînement du Modèle
   ↓
6. Évaluation et Validation
   ↓
7. Déploiement via API Flask
   ↓
8. Interface Web pour Prédictions
```

### Architecture du Modèle

**Type**: Classification Binaire (Spam vs Ham)

**Pipeline ML**:
1. **TfidfVectorizer** - Conversion du texte en vecteurs numériques basés sur TF-IDF
   - Tokenisation personnalisée
   - Filtrage des stop-words français
   - Extraction d'unigrammes et bigrammes

2. **Logistic Regression** - Modèle de classification
   - Algorithm linéaire et léger
   - Adapté pour les tâches de classification binaire
   - Explicitabilité des résultats

---

## 🤖 Méthodes de Machine Learning

### Algorithmes Utilisés

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Vectorisation des textes
   - Poids basé sur la fréquence des termes
   - Normalisation par la fréquence inverse dans le corpus

2. **Logistic Regression**
   - Modèle statistique linéaire
   - Sortie: Probabilité entre 0 et 1
   - Décision: Seuil à 0.5 (ajustable)

### Métriques d'Évaluation

- **Accuracy** - Proportion de prédictions correctes
- **Precision** - Proportion des vrais positifs parmi les positifs prédits
- **Recall** - Proportion des vrais positifs parmi les positifs réels
- **F1-Score** - Moyenne harmonique de Precision et Recall
- **Confusion Matrix** - Matrice de confusion pour analyse détaillée

### Split des Données

- **Ensemble d'entraînement**: 80%
- **Ensemble de test**: 20%
- **Stratification**: Respectée pour équilibrer les classes

---

## 📁 Datasets Utilisés

### Sources de Données

| Dataset | Description | Taille |
|---------|-------------|--------|
| **dataset_raw.csv** | Données brutes initiales avec messages en plusieurs langues | Donnée initiale |
| **dataset_french.csv** | Données filtrées - messages en français uniquement | Filtrée et nettoyée |
| **dataset.csv** | Dataset de référence pour validation | Validation |

### Structure des Données

```csv
text,labels
"Bonjour, how are you?",ham
"ACHETER MAINTENANT! CLIQUEZ ICI!!!",spam
...
```

### Prétraitement Appliqué

- ✅ Filtrage par langue (français uniquement)
- ✅ Suppression des doublons
- ✅ Nettoyage des caractères spéciaux
- ✅ Conversion en minuscules
- ✅ Suppression des stop-words français courants

---

## 🌐 Application Web Hébergée

### URL de Déploiement

**Application en Production**: [À remplir avec le lien Render]

*Exemple*: `https://spam-detector.onrender.com`

### Accès à l'Application

1. **Page d'accueil**: `/` - Interface utilisateur
2. **Endpoint API**: `/predict` (POST)
   
   **Request**:
   ```json
   {
     "text": "Votre message à vérifier"
   }
   ```
   
   **Response**:
   ```json
   {
     "prediction": 1,
     "text": "Votre message à vérifier",
     "label": "Spam",
     "confidence": 0.95
   }
   ```

### Instructions de Déploiement

Le projet est configuré pour être déployé sur **Render**:

1. Connecter le repository GitHub à Render
2. Configurer l'environnement Python
3. Installer les dépendances depuis `requirements.txt`
4. Builder avec `render-build.sh`
5. Démarrer le serveur avec Gunicorn

---

## 🔧 Installation Locale

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'Installation

```bash
# 1. Cloner le projet
git clone <votre-repository>
cd spam

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraîner le modèle (si nécessaire)
python scripts/02_train_model.py

# 4. Démarrer le serveur
python predict_server.py

# 5. Accéder à l'application
# Ouvrir http://localhost:5000 dans votre navigateur
```

---

## 📝 Scripts Disponibles

### `01_filter_french_data.py`
Filtre les données brutes pour conserver uniquement les messages en français.

### `02_train_model.py`
Entraîne le modèle Logistic Regression avec TF-IDF Vectorizer.

### `03_prediction_util.py`
Utilitaire de prédiction pour tester le modèle.

### `text_utils.py`
Fonctions utilitaires pour tokenisation et gestion des stop-words français.

### `predict_server.py`
Serveur Flask exposant l'API de prédiction et servant l'interface web.

---

## 📊 Fichiers Générés

Après l'entraînement du modèle:

- `scripts/spam_model.joblib` - Modèle entraîné (Logistic Regression)
- `scripts/vectorizer.joblib` - Vectorizer TF-IDF

---

## 📧 Contact & Support

Pour plus d'informations, visitez:
- 🌐 **Site ISPM**: www.ispm-edu.com
- 📱 **Équipe Projet**: [À compléter avec les coordonnées de contact]

---

## 📄 Licence

Ce projet est développé pour ISPM - Institut Supérieur Polytechnique de Madagascar.

---

**Dernière mise à jour**: Janvier 2026
