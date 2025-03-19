import pandas as pd
import numpy as np
import os
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ensure directories exist
os.makedirs('web/model', exist_ok=True)

print("Loading IPC data...")
df = pd.read_csv('ipc_sections.csv')

# Explore data
print(f"Dataset shape: {df.shape}")
print(f"Unique section count: {df['Section'].nunique()}")
print(f"Column names: {df.columns.tolist()}")

# Check for class imbalance
section_counts = df['Section'].value_counts()
print(f"Sections with only one example: {sum(section_counts == 1)}")
print(f"Most common sections: {section_counts.head(5).to_dict()}")

# Enhanced class imbalance handling
min_examples = 3  # Increased from 2 to ensure better representation
sections_to_keep = section_counts[section_counts >= min_examples].index
df_filtered = df[df['Section'].isin(sections_to_keep)]
print(f"Filtered data shape: {df_filtered.shape}")

# Plot class distribution
plt.figure(figsize=(14, 8))
sns.countplot(y=df_filtered['Section'], order=df_filtered['Section'].value_counts().index[:20])
plt.title('Distribution of Top 20 IPC Sections')
plt.tight_layout()
plt.savefig('web/model/class_distribution.png')
plt.close()

# Advanced text preprocessing
def advanced_clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Preserve important legal terms and patterns
    text = re.sub(r'section\s+(\d+[a-z]*)', r'section_\1', text)
    text = re.sub(r'ipc\s+(\d+[a-z]*)', r'ipc_\1', text)
    # Replace punctuation with space except for hyphens in legal codes
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create a more comprehensive feature set
df_filtered['section_numbers'] = df_filtered['Section'].str.extract(r'(\d+[A-Za-z]*)')
df_filtered['section_length'] = df_filtered['Section'].str.len()
df_filtered['offense_length'] = df_filtered['Offense'].astype(str).str.len()
df_filtered['description_length'] = df_filtered['Description'].astype(str).str.len()
df_filtered['punishment_length'] = df_filtered['Punishment'].astype(str).str.len()

# Create a weighted combined text field
df_filtered['combined_text'] = ''
for col, weight in [('Section', 2), ('Offense', 3), ('Description', 2), ('Punishment', 1)]:
    if col in df_filtered.columns:
        df_filtered['combined_text'] += ' ' + df_filtered[col].fillna('').astype(str) * weight

df_filtered['clean_text'] = df_filtered['combined_text'].apply(advanced_clean_text)

# Use legal domain-specific stopwords
legal_stopwords = ['the', 'of', 'and', 'to', 'in', 'a', 'is', 'be', 'that', 'by', 'for',
                  'with', 'as', 'on', 'at', 'which', 'or', 'from', 'an', 'this', 'are']

# Extract additional features
df_filtered['contains_section'] = df_filtered['clean_text'].str.contains('section').astype(int)
df_filtered['contains_imprisonment'] = df_filtered['clean_text'].str.contains('imprison|jail|custody').astype(int)
df_filtered['contains_fine'] = df_filtered['clean_text'].str.contains('fine|penalty|rupee').astype(int)

# Encode labels
label_encoder = LabelEncoder()
df_filtered['section_encoded'] = label_encoder.fit_transform(df_filtered['Section'])
num_classes = len(label_encoder.classes_)
print(f"Number of classes after filtering: {num_classes}")

# Split the data using stratified sampling
X_text = df_filtered['clean_text']
X_features = df_filtered[['section_length', 'offense_length', 'description_length', 
                         'punishment_length', 'contains_section', 'contains_imprisonment', 
                         'contains_fine']]
y = df_filtered['section_encoded']

X_train_text, X_test_text, X_train_features, X_test_features, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training examples: {len(X_train_text)}")
print(f"Testing examples: {len(X_test_text)}")

# Save dataset statistics
with open('web/model/training_info.txt', 'w') as f:
    f.write(f"Total examples: {len(df_filtered)}\n")
    f.write(f"Number of classes: {num_classes}\n")
    f.write(f"Training examples: {len(X_train_text)}\n")
    f.write(f"Testing examples: {len(X_test_text)}\n")
    f.write(f"Min examples per class: {min_examples}\n")
    f.write(f"Features: {X_features.columns.tolist()}\n")

# Create a more advanced feature extraction pipeline
text_features = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        stop_words=legal_stopwords
    )),
    ('svd', TruncatedSVD(n_components=300))  # Dimensionality reduction
])

# Create a column transformer to combine text and numerical features
preprocessor = ColumnTransformer([
    ('text', text_features, 'clean_text'),
    ('num', 'passthrough', ['section_length', 'offense_length', 'description_length', 
                          'punishment_length', 'contains_section', 'contains_imprisonment', 
                          'contains_fine'])
])

# Function to train and evaluate a model
def train_and_evaluate(model, name, X_train_text, X_train_features, y_train, X_test_text, X_test_features, y_test):
    print(f"\nTraining {name} model...")
    
    # Create a pipeline with the preprocessor and classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(pd.DataFrame({'clean_text': X_train_text}).join(X_train_features), y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(pd.DataFrame({'clean_text': X_test_text}).join(X_test_features))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, 
                               pd.DataFrame({'clean_text': X_train_text}).join(X_train_features), 
                               y_train, 
                               cv=5, 
                               scoring='accuracy')
    print(f"{name} Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return pipeline, accuracy

# Train multiple models
models = [
    (CalibratedClassifierCV(LinearSVC(C=2.0, class_weight='balanced', dual=False, max_iter=10000)), "Calibrated SVM"),
    (LogisticRegression(C=1.0, class_weight='balanced', max_iter=10000, solver='saga', multi_class='multinomial'), "Logistic Regression"),
    (MultinomialNB(alpha=0.01), "Naive Bayes"),
    (RandomForestClassifier(n_estimators=200, class_weight='balanced', min_samples_split=5), "Random Forest")
]

results = {}
for model, name in models:
    pipeline, accuracy = train_and_evaluate(model, name, X_train_text, X_train_features, y_train, X_test_text, X_test_features, y_test)
    results[name] = {'pipeline': pipeline, 'accuracy': accuracy}

# Select the best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['pipeline']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Create and save detailed evaluation metrics
y_pred = best_model.predict(pd.DataFrame({'clean_text': X_test_text}).join(X_test_features))
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('web/model/classification_report.csv')

# Plot confusion matrix for top classes
top_classes = section_counts.head(10).index
top_class_indices = [label_encoder.transform([cls])[0] for cls in top_classes]
y_test_top = np.array([y for y in y_test if y in top_class_indices])
y_pred_top = np.array([y_pred[i] for i, y in enumerate(y_test) if y in top_class_indices])

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_top, y_pred_top, labels=top_class_indices)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[label_encoder.inverse_transform([i])[0] for i in top_class_indices],
            yticklabels=[label_encoder.inverse_transform([i])[0] for i in top_class_indices])
plt.title('Confusion Matrix - Top 10 Classes')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('web/model/confusion_matrix.png')
plt.close()

# Save the best model
print(f"\nSaving the best model ({best_model_name})...")
joblib.dump(best_model, 'web/model/ipc_classifier.pkl')
joblib.dump(label_encoder, 'web/model/label_encoder.pkl')

# Create a more robust prediction function
with open('web/model/predict.py', 'w') as f:
    f.write("""
import re
import joblib
import pandas as pd
import numpy as np

# Load the model and label encoder
model = joblib.load('model/ipc_classifier.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

def advanced_clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Preserve important legal terms and patterns
    text = re.sub(r'section\s+(\\d+[a-z]*)', r'section_\\1', text)
    text = re.sub(r'ipc\s+(\\d+[a-z]*)', r'ipc_\\1', text)
    # Replace punctuation with space except for hyphens in legal codes
    text = re.sub(r'[^\\w\\s-]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def extract_features(text):
    '''
    Extract additional features from the text
    '''
    clean_text = advanced_clean_text(text)
    
    # Extract features
    features = {
        'section_length': len(text.split('section')[0]) if 'section' in text.lower() else 0,
        'offense_length': len(text),
        'description_length': len(text),
        'punishment_length': len(text),
        'contains_section': 1 if 'section' in text.lower() else 0,
        'contains_imprisonment': 1 if any(word in text.lower() for word in ['imprison', 'jail', 'custody']) else 0,
        'contains_fine': 1 if any(word in text.lower() for word in ['fine', 'penalty', 'rupee']) else 0
    }
    
    return clean_text, features

def predict_ipc_section(text, top_n=3):
    '''
    Predicts the IPC section for a given text.
    
    Args:
        text (str): The text to classify
        top_n (int): Number of top predictions to return
        
    Returns:
        dict: Dictionary containing the predicted sections and confidences
    '''
    # Clean the text and extract features
    clean_text, features = extract_features(text)
    
    # Create DataFrame with the required structure
    input_data = pd.DataFrame({
        'clean_text': [clean_text],
        'section_length': [features['section_length']],
        'offense_length': [features['offense_length']],
        'description_length': [features['description_length']],
        'punishment_length': [features['punishment_length']],
        'contains_section': [features['contains_section']],
        'contains_imprisonment': [features['contains_imprisonment']],
        'contains_fine': [features['contains_fine']]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    section = label_encoder.inverse_transform([prediction])[0]
    
    # Try to get probability if available
    try:
        probas = model.predict_proba(input_data)[0]
        # Get top N predictions
        top_indices = np.argsort(probas)[-top_n:][::-1]
        top_sections = label_encoder.inverse_transform(top_indices)
        top_probas = probas[top_indices]
        
        predictions = [
            {"section": section, "confidence": float(proba)} 
            for section, proba in zip(top_sections, top_probas)
        ]
    except:
        # If probabilities are not available, return just the top prediction
        predictions = [{"section": section, "confidence": None}]
    
    return {
        "top_prediction": predictions[0],
        "all_predictions": predictions
    }
""")

# Create a basic Flask API for the model
with open('web/app.py', 'w') as f:
    f.write("""
from flask import Flask, request, jsonify, render_template
from model.predict import predict_ipc_section
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    prediction = predict_ipc_section(text)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
""")

# Create a basic HTML template for the API
os.makedirs('web/templates', exist_ok=True)
with open('web/templates/index.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>IPC Section Predictor</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        textarea {
            width: 100%;
            min-height: 150px;
            margin-bottom: 10px;
            padding: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .prediction {
            font-weight: bold;
            color: #333;
        }
        .confidence {
            color: #666;
        }
        .alternative {
            margin-top: 10px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>IPC Section Predictor</h1>
    <p>Enter a description of the offense or legal text below:</p>
    
    <textarea id="input-text" placeholder="Enter text here..."></textarea>
    <button id="predict-button">Predict Section</button>
    
    <div id="result" class="result" style="display: none;">
        <h2>Prediction Result</h2>
        <div id="top-prediction">
            <p><span class="prediction">Predicted Section: </span><span id="section"></span></p>
            <p><span class="confidence">Confidence: </span><span id="confidence"></span></p>
        </div>
        <div id="alternatives">
            <h3>Alternative Predictions</h3>
            <div id="alt-predictions"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('predict-button').addEventListener('click', async () => {
            const text = document.getElementById('input-text').value;
            if (!text) {
                alert('Please enter some text');
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });
                
                const data = await response.json();
                
                // Display the result
                document.getElementById('section').textContent = data.top_prediction.section;
                document.getElementById('confidence').textContent = 
                    data.top_prediction.confidence ? 
                    `${(data.top_prediction.confidence * 100).toFixed(2)}%` : 
                    'Not available';
                
                // Display alternative predictions
                const altPredictions = document.getElementById('alt-predictions');
                altPredictions.innerHTML = '';
                
                if (data.all_predictions && data.all_predictions.length > 1) {
                    data.all_predictions.slice(1).forEach(pred => {
                        const predDiv = document.createElement('div');
                        predDiv.className = 'alternative';
                        predDiv.innerHTML = `
                            <p>Section: ${pred.section}</p>
                            <p>Confidence: ${pred.confidence ? 
                                `${(pred.confidence * 100).toFixed(2)}%` : 
                                'Not available'}</p>
                        `;
                        altPredictions.appendChild(predDiv);
                    });
                } else {
                    altPredictions.innerHTML = '<p>No alternative predictions available</p>';
                }
                
                document.getElementById('result').style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while making the prediction');
            }
        });
    </script>
</body>
</html>
""")

print("\nAdvanced model training and optimization complete!")
print(f"Final best model ({best_model_name}) accuracy: {best_accuracy:.4f}")
print("Model, API, and utilities saved to 'web/model/' directory")
print("To run the application, navigate to the 'web' directory and run 'python app.py'")