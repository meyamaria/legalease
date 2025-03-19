from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = Flask(__name__)
claude 
# Load ML components
model = tf.keras.models.load_model('web/model/legal_model.h5')

# Load TF-IDF Vectorizer
with open('web/model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load Tokenizer
with open('web/model/tokenizer.json', 'r') as f:
    tokenizer_config = json.load(f)

from tensorflow.keras.preprocessing.text import tokenizer_from_json
tokenizer = tokenizer_from_json(tokenizer_config)

# Load IPC Data
df = pd.read_csv('ipc_sections.csv')
df.columns = df.columns.str.strip()

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '').lower()
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    # Convert query to sequence
    query_seq = tokenizer.texts_to_sequences([query])
    query_padded = pad_sequences(query_seq, maxlen=100)

    # Predict using ML model
    prediction = model.predict(query_padded)
    best_match_idx = np.argmax(prediction)

    # Return result
    result = df.iloc[best_match_idx]
     return jsonify({
        "section": result.get('section', "N/A"),  # Use .get() to prevent KeyError
        "offense": result.get('offense', "N/A"),
        "description": result.get('description', "N/A"),
        "punishment": result.get('punishment', "N/A")
    })

if __name__ == '__main__':
    app.run(debug=True)
