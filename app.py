from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load ML components
print("Loading ML models and data...")

# Load TensorFlow model
model = tf.keras.models.load_model('web/model/legal_model.h5')

# Load TF-IDF Vectorizer
with open('web/model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load Tokenizer
with open('web/model/tokenizer.json', 'r') as f:
    tokenizer_config = json.load(f)
    
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_config))

# Load IPC Data
df = pd.read_csv('ipc_sections.csv')
df.columns = df.columns.str.strip()

# For TF-IDF fallback
tfidf_matrix = vectorizer.transform(df['Offense'] + ' ' + df['Description'])

@app.route('/search', methods=['POST'])
def search():
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
        
        # Return top 3 similar sections if available
        related_sections = []
        if len(df) > 1:
            # Get top 3 predictions from neural model
            top_indices = np.argsort(prediction)[-3:][::-1]
            for idx in top_indices:
                if idx != best_match_idx:  # Don't include the best match again
                    row = df.iloc[idx]
                    related_sections.append({
                        "section": row.get('Section', "N/A"),
                        "offense": row.get('Offense', "N/A"),
                    })
        
        return jsonify({
            "section": result.get('Section', "N/A"),
            "offense": result.get('Offense', "N/A"),
            "description": result.get('Description', "N/A"),
            "punishment": result.get('Punishment', "N/A"),
            "confidence": float(confidence),
            "related_sections": related_sections
        })
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)