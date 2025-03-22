import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical

# 1. Load and prepare data
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna("")
    
    # Create a combined text field for training
    df['combined_text'] = df['Offense'] + ' ' + df['Description']
    
    # Assign numerical labels (we'll predict section indices)
    df['label'] = range(len(df))
    
    return df

# 2. Text preprocessing
def preprocess_data(df, max_features=5000, max_seq_length=100):
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['combined_text'])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['combined_text'])
    
    # Pad sequences to ensure uniform length
    X = pad_sequences(sequences, maxlen=max_seq_length)
    
    # Convert labels to categorical
    y = to_categorical(df['label'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, tokenizer

# 3. Build model
def build_model(max_features, max_seq_length, num_classes):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_seq_length))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

# 4. Train TF-IDF model as backup
def train_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, tfidf_matrix

# 5. Save all model components
def save_models(model, tokenizer, vectorizer, output_dir='web/model'):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save keras model
    model.save(os.path.join(output_dir, 'legal_model.h5'))
    
    # Save tokenizer
    tokenizer_json = tokenizer.to_json()
    with open(os.path.join(output_dir, 'tokenizer.json'), 'w') as f:
        f.write(tokenizer_json)
    
    # Save TF-IDF vectorizer
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"All models saved to {output_dir}")

# Main execution
if __name__ == "__main__":
    file_path = "ipc_sections.csv"
    max_features = 5000  # Vocabulary size
    max_seq_length = 100  # Maximum text length
    
    # Load and prepare data
    df = prepare_data(file_path)
    num_classes = len(df)  # Number of IPC sections
    
    # Preprocess data
    X_train, X_test, y_train, y_test, tokenizer = preprocess_data(
        df, max_features, max_seq_length
    )
    
    # Build and train neural model
    print("Training neural model...")
    model = build_model(max_features, max_seq_length, num_classes)
    model.summary()
    
    # Train the model with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    
    # Train TF-IDF model as backup
    print("Training TF-IDF model...")
    vectorizer, _ = train_tfidf(df)
    
    # Save all models
    save_models(model, tokenizer, vectorizer)
    
    # Also save the IPC data for reference
