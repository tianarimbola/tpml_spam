# Détection de Spam en Français - Guide Complet

## 📋 Vue d'ensemble

Ce projet crée un modèle de Machine Learning pour détecter les messages spam en français avec un score de confiance.

**Données**: Dataset filtré en français uniquement  
**Modèle**: Logistic Regression + TF-IDF Vectorizer  
**Framework**: scikit-learn

---

## 🚀 Installation et Utilisation

### 1. **Installer les dépendances**

```bash
pip install -r scripts/requirements.txt
```

### 2. **Étape 1 - Filtrer les données en français** ⚙️

```bash
python scripts/01_filter_french_data.py
```

**Sortie attendue:**
- Créé: `scripts/dataset_french.csv`
- Affiche le nombre de messages spam/ham en français
- Montre des exemples de chaque catégorie

### 3. **Étape 2 - Entraîner le modèle** 🤖

```bash
python scripts/02_train_model.py
```

**Sortie attendue:**
- Entraîne le modèle sur 80% des données
- Crée: `scripts/spam_model.joblib`
- Crée: `scripts/vectorizer.joblib`
- Affiche les métriques d'évaluation (Accuracy, Precision, Recall, F1-Score)
- Montre des tests manuels d'exemple

### 4. **Étape 3 - Tester le modèle** ✅

```bash
python scripts/03_prediction_util.py
```

**Sortie attendue:**
- Tests le modèle sur 5 messages d'exemple
- Affiche le résultat (SPAM/HAM) et le pourcentage de confiance

---

## 📁 Fichiers Générés

| Fichier | Description |
|---------|-------------|
| `dataset_french.csv` | Dataset filtré (seulement français) |
| `spam_model.joblib` | Modèle ML entraîné |
| `vectorizer.joblib` | TF-IDF Vectorizer |

---

## 💻 Utilisation en Production (API)

Voir le fichier `03_prediction_util.py` pour la classe `SpamDetector` qui peut être importée.

**Exemple:**
```python
from prediction_util import SpamDetector

detector = SpamDetector()
result = detector.predict("Votre message ici")
print(result['is_spam'])  # True ou False
print(result['confidence_percent'])  # 0-100%
```

---

## 📊 Métriques du Modèle

Le script d'entraînement affiche:
- **Accuracy**: Pourcentage global de prédictions correctes
- **Precision**: Parmi les messages marqués SPAM, combien sont vrais spam
- **Recall**: Combien de vrais spams le modèle détecte
- **F1-Score**: Balance entre Precision et Recall

---

## 📝 Notes

- Le dataset original contenait plusieurs langues, seuls les messages en français sont utilisés
- Le modèle utilise TF-IDF avec n-grams (1-2 mots) pour capturer les patterns de spam
- Class weight 'balanced' pour gérer le déséquilibre possible spam/ham
- Strip accents activé pour normaliser les caractères accentués en français

---

## 🎯 Prochaines Étapes

1. ✅ Entraîner le modèle localement
2. 🔄 Intégrer dans l'API Next.js (Route Handler)
3. 🎨 Créer l'interface web React
4. 📱 Tester avec des messages réels
5. 📈 Améliorer si nécessaire avec plus de données
