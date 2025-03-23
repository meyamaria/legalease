# train_model.py
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
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.utils import to_categorical

# Create directories
os.makedirs('web/model', exist_ok=True)

# 1. Load and prepare data
print("Loading data...")
df = pd.read_csv("ipc_sections.csv")
df = df.fillna("")

# Create a combined text field for training
df['combined_text'] = df['Offense'] + ' ' + df['Description'] 

# Assign numerical labels
df['label'] = range(len(df))

# 2. Text preprocessing
print("Preprocessing text...")
max_features = 5000  # Vocabulary size
max_seq_length = 100  # Maximum text length

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

# 3. Build and train model
print("Building and training model...")
model = Sequential()
model.add(Embedding(max_features, 64, input_length=max_seq_length))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(df), activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(
    X_train, y_train,
    epochs=10,  # Reduced for faster training
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 4. Train TF-IDF model as backup
print("Training TF-IDF model...")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# 5. Save all model components
print("Saving models...")
model.save('web/model/legal_model.h5')

# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open('web/model/tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

# Save TF-IDF vectorizer
with open('web/model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save a copy of the data for reference
df.to_csv('web/model/ipc_data.csv', index=False)

print("Training complete. All models saved to web/model/")