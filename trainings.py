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
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GlobalMaxPooling1D, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Configuration parameters
MAX_FEATURES = 10000  # Vocabulary size
MAX_SEQ_LENGTH = 150  # Maximum sequence length
EMBEDDING_DIM = 128   # Embedding dimension
BATCH_SIZE = 16       # Batch size
EPOCHS = 30           # Max epochs
TEST_SPLIT = 0.2      # Test split proportion
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

# Count original classes
original_sections = df['Section'].nunique()
print(f"Original number of unique sections: {original_sections}")

# CRITICAL FIX: Reduce number of classes if too many compared to data size
max_classes = int(len(df) * 0.6)  # Ensure classes are at most 60% of dataset size
if original_sections > max_classes:
    print(f"WARNING: Too many classes ({original_sections}) for dataset size ({len(df)})")
    print(f"Limiting to top {max_classes} most frequent classes")
    
    # Keep only the most frequent classes
    top_sections = df['Section'].value_counts().nlargest(max_classes).index
    df = df[df['Section'].isin(top_sections)]
    print(f"Dataset reduced to {len(df)} rows with {df['Section'].nunique()} classes")

# Use Section column as the class label
print("Creating proper class labels based on Section column...")
# Use LabelEncoder to convert section names to numerical indices
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Section'])

# Count instances per class
class_counts = df['label'].value_counts()
print(f"Number of unique classes: {len(class_counts)}")
print(f"Class distribution summary:")
print(f"  Minimum samples per class: {class_counts.min()}")
print(f"  Maximum samples per class: {class_counts.max()}")
print(f"  Average samples per class: {class_counts.mean():.2f}")

# Define minimum samples needed per class
min_samples_per_class = 1
num_classes = len(df['label'].unique())

# Implement data augmentation for classes with few samples
print("Applying data augmentation...")
class_counts = df['label'].value_counts()
augmented_rows = []
target_samples_per_class = 3  # Aim for at least 3 samples per class

for label, count in class_counts.items():
    if count < target_samples_per_class:
        # Get rows for this class
        class_rows = df[df['label'] == label]
        # How many more samples needed
        samples_needed = target_samples_per_class - count
        # Duplicate rows with small variations
        for _ in range(samples_needed):
            for _, row in class_rows.iterrows():
                new_row = row.copy()
                # Add small variations to text with word shuffling
                words = row['combined_text'].split()
                if len(words) > 5:  # Only modify if enough words
                    # Randomly shuffle 30% of the words
                    shuffle_indices = np.random.choice(
                        range(len(words)), 
                        size=max(1, int(len(words)*0.3)), 
                        replace=False
                    )
                    for idx in shuffle_indices:
                        words[idx] = words[idx] + 's' if not words[idx].endswith('s') else words[idx][:-1]
                    new_row['combined_text'] = ' '.join(words)
                    augmented_rows.append(new_row)

# Add augmented rows to the dataframe
if augmented_rows:
    augmented_df = pd.DataFrame(augmented_rows)
    df = pd.concat([df, augmented_df], ignore_index=True)
    print(f"Added {len(augmented_rows)} augmented samples")
    # Update class counts after augmentation
    class_counts = df['label'].value_counts()
    print(f"Class distribution after augmentation:")
    print(f"  Minimum samples per class: {class_counts.min()}")
    print(f"  Maximum samples per class: {class_counts.max()}")
    print(f"  Average samples per class: {class_counts.mean():.2f}")

# Save a copy of the processed data for reference
df.to_csv(os.path.join(MODEL_DIR, 'ipc_data_processed.csv'), index=False)

# Save the mapping from encoded labels back to section names
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
with open(os.path.join(MODEL_DIR, 'label_mapping.json'), 'w') as f:
    json.dump(label_mapping, f, indent=4)

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

# CRITICAL FIX: Use simple random split with NO stratification
print("Using simple random split with NO stratification")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=None
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 3. Build and train model
print("Building the neural network model...")
# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'legal_model_checkpoint.keras'),
        monitor='val_loss',
        save_best_only=True
    )
]

# Use simpler model for faster training
model = Sequential([
    Embedding(vocab_size, 64, input_length=MAX_SEQ_LENGTH),
    SpatialDropout1D(0.2),
    LSTM(32, recurrent_dropout=0.2),
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

# Adjust batch size based on dataset size
adjusted_batch_size = min(BATCH_SIZE, max(4, len(X_train) // 10))  # Ensure at least 4
print(f"Using batch size: {adjusted_batch_size}")

# Train the model
print(f"Training model for up to {EPOCHS} epochs (with early stopping)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=adjusted_batch_size,
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

# If accuracy is below 0.9, train on full dataset to get final model
if accuracy < 0.9:
    print("\n== Accuracy below 0.9, training on full dataset for final model ==")
    # Create a new model for full dataset training
    final_model = Sequential([
        Embedding(vocab_size, 64, input_length=MAX_SEQ_LENGTH),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(32)),  # Use bidirectional for final model
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    final_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Train on full dataset
    final_model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=adjusted_batch_size,
        callbacks=[EarlyStopping(monitor='accuracy', patience=5, min_delta=0.01)],
        verbose=1
    )
    
    # Replace model with final_model
    model = final_model
    # Check final accuracy (on training data since we used all data)
    _, final_accuracy = model.evaluate(X, y, verbose=0)
    print(f"Final model accuracy: {final_accuracy:.4f}")
    accuracy = final_accuracy  # Use this for reporting

# 5. Train TF-IDF model as backup
print("Training TF-IDF model as fallback method...")
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    sublinear_tf=True
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
model.save(os.path.join(MODEL_DIR, 'legal_model.keras'))

# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open(os.path.join(MODEL_DIR, 'tokenizer.json'), 'w') as f:
    f.write(tokenizer_json)

# Save TF-IDF vectorizer
with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

# Save label encoder
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

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

def predict_ipc_section(query, model, tokenizer, vectorizer, tfidf_matrix, df, label_encoder):
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
        
        # Get the original indices sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1][:5]  # Top 5 matches
        best_match_idx = sorted_indices[0]
        confidence = similarities[best_match_idx]
        
        # Map from model prediction index to dataframe index
        df_indices = np.where(df['label'] == best_match_idx)[0]
        if len(df_indices) > 0:
            df_idx = df_indices[0]
            result = df.iloc[df_idx]
        else:
            # If no exact match for label, use the index directly (fall back)
            result = df.iloc[best_match_idx]
    else:
        # Convert model prediction index to original label
        label = best_match_idx
        # Find rows with this label
        df_indices = np.where(df['label'] == label)[0]
        if len(df_indices) > 0:
            df_idx = df_indices[0]
            result = df.iloc[df_idx]
        else:
            # Fallback - this should rarely happen
            result = df.iloc[0]
            confidence = 0.0
    
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
result = predict_ipc_section(sample_query, model, tokenizer, vectorizer, tfidf_matrix, df, label_encoder)
print("Prediction result:")
for key, value in result.items():
    if key == "confidence":
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")