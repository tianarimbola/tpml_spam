from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from text_utils import simple_tokenizer, french_stop_words

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'scripts/spam_model.joblib')
VECT_PATH = os.path.join(BASE_DIR, 'scripts/vectorizer.joblib')


try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    loaded = True
    print('Modèle et vectorizer chargés.')
except Exception as e:
    print(f"Erreur lors du chargement du modèle/vectorizer: {e}")
    model = None
    vectorizer = None
    loaded = False


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not loaded:
        return jsonify({'error': 'Modèle non chargé'}), 500

    data = request.get_json(silent=True) or {}
    text = data.get('text') or request.form.get('text')

    if not text:
        return jsonify({'error': 'Le champ "text" est requis.'}), 400

    try:
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]

        is_spam = bool(prediction)
        confidence = float(probabilities[1] if prediction == 1 else probabilities[0])

        return jsonify({
            'message': text,
            'is_spam': is_spam,
            'label': 'SPAM' if is_spam else 'HAM',
            'confidence': confidence,
            'confidence_percent': confidence * 100,
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
