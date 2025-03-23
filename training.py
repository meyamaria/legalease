# improved_train_model.py
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration parameters
MAX_FEATURES = 10000  # Vocabulary size (increased from original)
MAX_SEQ_LENGTH = 150  # Maximum sequence length (increased from original)
EMBEDDING_DIM = 128   # Embedding dimension (increased from original)
BATCH_SIZE = 32       # Batch size for training
EPOCHS = 20           # Maximum number of epochs
TEST_SPLIT = 0.2      # Proportion of data to use for testing
MODEL_DIR = 'web/model'

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Model directory ready: {MODEL_DIR}")

# 1. Load and prepare data
print("Loading IPC dataset...")
try:
    df = pd.read_csv("ipc_sections.csv")
    # Print dataset information
    print(f"Dataset loaded: {len(df)} rows")
    print(f"Columns available: {df.columns.tolist()}")
except FileNotFoundError:
    print("Error: ipc_sections.csv file not found. Please ensure the file exists in the current directory.")
    exit(1)

# Check for required columns
required_columns = ["Section", "Offense", "Description", "Punishment"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: The following required columns are missing: {missing_columns}")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Clean and preprocess data
print("Preprocessing dataset...")
# Fill missing values
df = df.fillna("")

# Create a combined text field for training (including all relevant columns)
df['combined_text'] = df['Section'] + ' ' + df['Offense'] + ' ' + df['Description'] + ' ' + df['Punishment']
df['combined_text'] = df['combined_text'].str.lower()  # Convert to lowercase

# Assign numerical labels
df['label'] = range(len(df))
num_classes = len(df)
print(f"Number of unique IPC sections/classes: {num_classes}")

# Save a copy of the processed data for reference
df.to_csv(os.path.join(MODEL_DIR, 'ipc_data_processed.csv'), index=False)

# 2. Text preprocessing
print("Tokenizing and preparing sequences...")
# Initialize tokenizer
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token='<OOV>')
tokenizer.fit_on_texts(df['combined_text'])

# Get vocabulary size
vocab_size = min(MAX_FEATURES, len(tokenizer.word_index) + 1)
print(f"Vocabulary size: {vocab_size}")

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['combined_text'])

# Pad sequences to ensure uniform length
X = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)

# Convert labels to categorical representation
y = to_categorical(df['label'], num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y if len(df) > 10 else None
)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 3. Build and train model
print("Building the neural network model...")
# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'legal_model_checkpoint.h5'),
        monitor='val_loss',
        save_best_only=True
    )
]

# Build a more sophisticated model
model = Sequential([
    # Embedding layer
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH),
    
    # Bidirectional LSTM layer
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.4),
    
    # Second LSTM layer
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    
    # Dense layers
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model
print(f"Training model for up to {EPOCHS} epochs (with early stopping)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 4. Evaluate the model
print("Evaluating the model...")
# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print detailed metrics for the first few classes to avoid overwhelming output
print("\nClassification Report (first 5 classes):")
target_names = [f"Class {i}" for i in range(min(5, num_classes))]
print(classification_report(
    y_true, y_pred, 
    labels=range(min(5, num_classes)), 
    target_names=target_names, 
    zero_division=0
))

# 5. Train TF-IDF model as backup
print("Training TF-IDF model as fallback method...")
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),  # Include both unigrams and bigrams
    sublinear_tf=True    # Apply sublinear tf scaling (log scaling)
)
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# 6. Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
print(f"Training history plot saved to {os.path.join(MODEL_DIR, 'training_history.png')}")

# 7. Save all model components
print("Saving all model components...")
# Save the main model
model.save(os.path.join(MODEL_DIR, 'legal_model.h5'))

# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open(os.path.join(MODEL_DIR, 'tokenizer.json'), 'w') as f:
    f.write(tokenizer_json)

# Save TF-IDF vectorizer
with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

# Save training configuration for reference
config = {
    'max_features': MAX_FEATURES,
    'max_seq_length': MAX_SEQ_LENGTH,
    'embedding_dim': EMBEDDING_DIM,
    'num_classes': num_classes,
    'test_accuracy': float(accuracy)
}
with open(os.path.join(MODEL_DIR, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=4)

print("\n===== Training complete =====")
print(f"All models and data saved to {MODEL_DIR}/")
print("You can now run the Flask application (app.py) to use these models.")

# 8. Generate a sample query test to verify the model works correctly
print("\nTesting model with a sample query...")

def predict_ipc_section(query, model, tokenizer, vectorizer, tfidf_matrix, df):
    # Preprocess query
    query = query.lower()
    
    # METHOD 1: Neural network prediction
    query_seq = tokenizer.texts_to_sequences([query])
    query_padded = pad_sequences(query_seq, maxlen=MAX_SEQ_LENGTH)
    prediction = model.predict(query_padded)[0]
    confidence = np.max(prediction)
    best_match_idx = np.argmax(prediction)
    
    # METHOD 2: TF-IDF Fallback if confidence is low
    if confidence < 0.5:  # Threshold for confidence
        # Use TF-IDF similarity as fallback
        query_vec = vectorizer.transform([query])
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]
    
    # Get the matching result
    result = df.iloc[best_match_idx]
    
    return {
        "section": result.get('Section', "N/A"),
        "offense": result.get('Offense', "N/A"),
        "description": result.get('Description', "N/A"),
        "punishment": result.get('Punishment', "N/A"),
        "confidence": float(confidence)
    }

# Test with a sample query - modify this to match something in your dataset
sample_query = "theft of property"  # Replace with a query related to your IPC data
print(f"Sample query: '{sample_query}'")
result = predict_ipc_section(sample_query, model, tokenizer, vectorizer, tfidf_matrix, df)
print("Prediction result:")
for key, value in result.items():
    if key == "confidence":
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")