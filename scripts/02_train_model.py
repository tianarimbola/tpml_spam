"""
Script pour entraîner le modèle de détection de spam en français
Utilise: TfidfVectorizer + Logistic Regression
"""
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import re
from text_utils import simple_tokenizer, french_stop_words
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

print("=" * 60)
print("ENTRAÎNEMENT DU MODÈLE DE DÉTECTION DE SPAM EN FRANÇAIS")
print("=" * 60)

# 1. Charger les données
print("\n1. Chargement des données...")
df = pd.read_csv('dataset_french.csv')
print(f"   Total de messages: {len(df)}")
print(f"   Distribution: {dict(df['labels'].value_counts())}")

# 2. Préparation des données
print("\n2. Préparation des données...")
X = df['text'].values
y = df['labels'].values

# Convertir les labels en nombres (0 = ham, 1 = spam)
y_binary = (y == 'spam').astype(int)

print(f"   Spam: {sum(y_binary)} messages")
print(f"   Ham (non-spam): {len(y_binary) - sum(y_binary)} messages")

# 3. Split train/test
print("\n3. Séparation train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"   Train: {len(X_train)} messages")
print(f"   Test: {len(X_test)} messages")

# 4. Vectorisation TF-IDF
print("\n4. Prétraitement et vectorisation des textes (tokenisation, stop-words, BOW/ngram, TF-IDF)...")

# Tokenizer and stop-words are imported from scripts.text_utils

# Pipeline: CountVectorizer (Bag-of-Words + n-grams + stop words FR) -> TF-IDF
vectorizer = Pipeline([
    ('vect', CountVectorizer(
        tokenizer=simple_tokenizer,
        stop_words=french_stop_words,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        max_features=5000,
        lowercase=True
    )),
    ('tfidf', TfidfTransformer())
])

# Fit transform train and transform test
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"   Nombre de features: {X_train_tfidf.shape[1]}")

# 5. Entraînement du modèle
print("\n5. Entraînement du modèle Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

# Entraînement sur la représentation TF-IDF
model.fit(X_train_tfidf, y_train)
print("   ✓ Modèle entraîné!")

# 6. Évaluation
print("\n6. Évaluation du modèle...")
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Matrice de confusion
print("\n   Matrice de confusion:")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")

# 7. Sauvegarder le modèle et le vectorizer
print("\n7. Sauvegarde du modèle...")
base_dir = os.path.dirname(__file__)
joblib.dump(model, os.path.join(base_dir, 'spam_model.joblib'))
# Sauvegarde du pipeline de vectorisation (CountVectorizer + TF-IDF)
joblib.dump(vectorizer, os.path.join(base_dir, 'vectorizer.joblib'))
print("   ✓ Modèle sauvegardé: spam_model.joblib")
print("   ✓ Vectorizer sauvegardé: vectorizer.joblib")

# 8. Test manuel
print("\n" + "=" * 60)
print("TEST MANUEL DU MODÈLE")
print("=" * 60)

def predict_spam(text):
    """Fonction pour prédire si un message est spam"""
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf)[0]
    
    is_spam = "SPAM ⚠️" if prediction == 1 else "HAM ✓"
    confidence_score = confidence[1] if prediction == 1 else confidence[0]
    
    return {
        'message': text,
        'prediction': is_spam,
        'confidence': confidence_score
    }

print("\n" + "=" * 60)
print("✓ ENTRAÎNEMENT TERMINÉ!")
print("=" * 60)
