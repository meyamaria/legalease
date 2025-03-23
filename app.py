from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import json
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Check if model directory exists
if not os.path.exists('web/model'):
    os.makedirs('web/model', exist_ok=True)
    print("Model directory created. Please run train_model.py first to train and save models.")

# Global variables
model = None
vectorizer = None
tokenizer = None
df = None
tfidf_matrix = None

# Load models if they exist
def load_models():
    global model, vectorizer, tokenizer, df, tfidf_matrix
    
    try:
        print("Loading ML models and data...")
        
        # Load TensorFlow model
        if os.path.exists('web/model/legal_model.h5'):
            model = tf.keras.models.load_model('web/model/legal_model.h5')
        else:
            print("Model file not found. Please run train_model.py first.")
            return False
            
        # Load TF-IDF Vectorizer
        if os.path.exists('web/model/tfidf_vectorizer.pkl'):
            with open('web/model/tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
        else:
            print("Vectorizer file not found. Please run train_model.py first.")
            return False
            
        # Load Tokenizer
        if os.path.exists('web/model/tokenizer.json'):
            with open('web/model/tokenizer.json', 'r') as f:
                tokenizer_config = json.load(f)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_config))
        else:
            print("Tokenizer file not found. Please run train_model.py first.")
            return False
            
        # Load IPC Data
        if os.path.exists('ipc_sections.csv'):
            df = pd.read_csv('ipc_sections.csv')
            df.columns = df.columns.str.strip()
            
            # For TF-IDF fallback
            if vectorizer is not None:
                tfidf_matrix = vectorizer.transform(df['Offense'] + ' ' + df['Description'])
        else:
            print("IPC sections data not found.")
            return False
            
        print("All models and data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

# Try to load models at startup
models_loaded = load_models()

@app.route('/search', methods=['POST'])
def search():
    global models_loaded
    
    # Check if models are loaded
    if not models_loaded:
        models_loaded = load_models()
        if not models_loaded:
            return jsonify({"error": "ML models not loaded. Please run train_model.py first."}), 500
    
    data = request.json
    query = data.get('query', '').lower()
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    try:
        # METHOD 1: Neural network prediction
        # Convert query to sequence
        query_seq = tokenizer.texts_to_sequences([query])
        query_padded = pad_sequences(query_seq, maxlen=100)
        
        # Predict using ML model
        prediction = model.predict(query_padded)[0]
        confidence = np.max(prediction)
        best_match_idx = np.argmax(prediction)
        
        # METHOD 2: TF-IDF Fallback if confidence is low
        if confidence < 0.5:  # Threshold for confidence
            # Use TF-IDF similarity as fallback
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            best_match_idx = np.argmax(similarities)
            confidence = similarities[best_match_idx]
        
        # Get the matching result
        result = df.iloc[best_match_idx]
        
        return jsonify({
            "section": result.get('Section', "N/A"),
            "offense": result.get('Offense', "N/A"),
            "description": result.get('Description', "N/A"),
            "punishment": result.get('Punishment', "N/A"),
            "confidence": float(confidence)
        })
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)