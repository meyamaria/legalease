# test_ipc_queries.py
# This script lets you test your trained model with sample queries

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import json
import os
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
MODEL_DIR = 'web/model'

def load_models():
    """Load all necessary model components"""
    print("Loading models and data...")
    
    # Load TensorFlow model
    model_path = os.path.join(MODEL_DIR, 'legal_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✓ Neural network model loaded")
    
    # Load TF-IDF Vectorizer
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✓ TF-IDF vectorizer loaded")
    
    # Load Tokenizer
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    with open(tokenizer_path, 'r') as f:
        tokenizer_config = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_config))
    print("✓ Tokenizer loaded")
    
    # Load IPC Data
    data_path = os.path.join(MODEL_DIR, 'ipc_data_processed.csv')
    if not os.path.exists(data_path):
        # Try the original data path
        data_path = "ipc_sections.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"IPC data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    print(f"✓ IPC data loaded with {len(df)} sections")
    
    # Load configuration
    config_path = os.path.join(MODEL_DIR, 'training_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Training configuration loaded: {config}")
    else:
        config = {'max_seq_length': 100}  # Default if not found
    
    # Prepare TF-IDF matrix for fallback method
    if 'combined_text' in df.columns:
        text_field = 'combined_text'
    else:
        # Fallback to combining fields
        text_field = 'Offense'
        if 'Description' in df.columns:
            text_field += ' Description'
    
    tfidf_matrix = vectorizer.transform(df[text_field])
    print(f"✓ TF-IDF matrix prepared with shape {tfidf_matrix.shape}")
    
    return model, tokenizer, vectorizer, tfidf_matrix, df, config

def predict_ipc_section(query, model, tokenizer, vectorizer, tfidf_matrix, df, config):
    """Predict IPC section for a given query"""
    start_time = time.time()
    
    # Preprocess query
    query = query.lower()
    
    # METHOD 1: Neural network prediction
    query_seq = tokenizer.texts_to_sequences([query])
    max_seq_length = config.get('max_seq_length', 100)
    query_padded = pad_sequences(query_seq, maxlen=max_seq_length)
    
    nn_start = time.time()
    prediction = model.predict(query_padded, verbose=0)[0]
    nn_time = time.time() - nn_start
    
    confidence = np.max(prediction)
    best_match_idx = np.argmax(prediction)
    method = "Neural Network"
    
    # METHOD 2: TF-IDF Fallback if confidence is low
    if confidence < 0.5:  # Threshold for confidence
        tfidf_start = time.time()
        
        # Use TF-IDF similarity as fallback
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        tfidf_best_idx = np.argmax(similarities)
        tfidf_confidence = similarities[tfidf_best_idx]
        
        tfidf_time = time.time() - tfidf_start
        
        # If TF-IDF gives better confidence, use it
        if tfidf_confidence > confidence:
            best_match_idx = tfidf_best_idx
            confidence = tfidf_confidence
            method = "TF-IDF (Fallback)"
    else:
        tfidf_time = 0
    
    # Get the matching result
    result = df.iloc[best_match_idx]
    
    # Format results
    response = {
        "section": result.get('Section', "N/A"),
        "offense": result.get('Offense', "N/A"),
        "description": result.get('Description', "N/A"),
        "punishment": result.get('Punishment', "N/A"),
        "confidence": float(confidence),
        "method": method,
        "processing_time": time.time() - start_time
    }
    
    return response

def main():
    """Main function to test the model with queries"""
    try:
        # Load models
        model, tokenizer, vectorizer, tfidf_matrix, df, config = load_models()
        
        # Test queries
        print("\n=== IPC Query Testing Tool ===")
        print("Type your legal queries below, or 'exit' to quit.\n")
        
        while True:
            query = input("\nEnter legal query: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            if not query.strip():
                continue
            
            # Get prediction
            result = predict_ipc_section(query, model, tokenizer, vectorizer, tfidf_matrix, df, config)
            
            # Display results
            print("\n----- Results -----")
            print(f"Query: '{query}'")
            print(f"Method: {result['method']} (confidence: {result['confidence']:.4f})")
            print(f"Processing time: {result['processing_time']*1000:.2f} ms")
            print("\nMatched IPC Section:")
            print(f"  Section: {result['section']}")
            print(f"  Offense: {result['offense']}")
            print(f"  Description: {result['description'][:200]}..." if len(result.get('description', '')) > 200 
                  else f"  Description: {result['description']}")
            print(f"  Punishment: {result['punishment'][:200]}..." if len(result.get('punishment', '')) > 200 
                  else f"  Punishment: {result['punishment']}")
        
        print("\nExiting IPC Query Testing Tool.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()