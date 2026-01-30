"""
Utilitaire pour faire des prédictions avec le modèle entraîné
Ce fichier peut être utilisé pour tester localement ou être importé
"""
import joblib
import numpy as np

class SpamDetector:
    """Classe pour détection de spam en français"""
    
    def __init__(self, model_path='spam_model.joblib', 
                 vectorizer_path='vectorizer.joblib'):
        """
        Initialise le détecteur de spam
        
        Args:
            model_path: chemin vers le modèle sauvegardé
            vectorizer_path: chemin vers le vectorizer sauvegardé
        """
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.is_loaded = True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.is_loaded = False
    
    def predict(self, text):
        """
        Prédit si un message est spam ou non
        
        Args:
            text (str): Le message à analyser
            
        Returns:
            dict: {
                'message': str,
                'is_spam': bool,
                'confidence': float (0-1),
                'confidence_percent': float (0-100)
            }
        """
        if not self.is_loaded:
            return {
                'message': text,
                'is_spam': None,
                'confidence': None,
                'confidence_percent': None,
                'error': 'Modèle non chargé'
            }
        
        try:
            # Vectorisation du texte
            text_tfidf = self.vectorizer.transform([text])
            
            # Prédiction
            prediction = self.model.predict(text_tfidf)[0]
            probabilities = self.model.predict_proba(text_tfidf)[0]
            
            # Confiance: probabilité de la classe prédite
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            return {
                'message': text,
                'is_spam': bool(prediction),
                'confidence': float(confidence),
                'confidence_percent': float(confidence * 100),
                'spam_probability': float(probabilities[1]),
                'ham_probability': float(probabilities[0])
            }
        except Exception as e:
            return {
                'message': text,
                'is_spam': None,
                'confidence': None,
                'confidence_percent': None,
                'error': str(e)
            }
    
    def batch_predict(self, texts):
        """
        Prédit plusieurs messages à la fois
        
        Args:
            texts (list): Liste de messages à analyser
            
        Returns:
            list: Liste de dictionnaires avec les prédictions
        """
        return [self.predict(text) for text in texts]


# Exemple d'utilisation
if __name__ == "__main__":
    # Ce module est conçu pour être importé par une application (ex: serveur Flask).
    # Le bloc d'exemples interactifs a été retiré car le front-end fournit le formulaire.
    pass
