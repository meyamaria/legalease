# data_preprocessing.py
# This script helps prepare and clean your IPC dataset before training

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download necessary NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    return ' '.join(cleaned_tokens)

# Load data
print("Loading IPC dataset...")
try:
    df = pd.read_csv("ipc_sections.csv")
    print(f"Dataset loaded: {len(df)} rows")
    print(f"Columns available: {df.columns.tolist()}")
except FileNotFoundError:
    print("Error: ipc_sections.csv file not found. Please ensure the file exists in the current directory.")
    exit(1)

# Check for required columns
required_columns = ["Section", "Offense", "Description", "Punishment"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Warning: The following recommended columns are missing: {missing_columns}")
    print(f"Available columns: {df.columns.tolist()}")

# Basic cleaning
print("Cleaning and preprocessing data...")
# Fill missing values
df = df.fillna("")

# Strip whitespace from column names and values
df.columns = df.columns.str.strip()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()

# Create a duplicate with cleaned text for enhanced search
print("Creating enhanced text features...")
for col in ["Offense", "Description", "Punishment"]:
    if col in df.columns:
        df[f'cleaned_{col}'] = df[col].apply(clean_text)

# Create combined text fields
df['combined_text'] = ''
for col in required_columns:
    if col in df.columns:
        df['combined_text'] += df[col] + ' '

# Create cleaned combined text
df['cleaned_combined_text'] = clean_text(df['combined_text'])

# Add section number extraction for better matching
if 'Section' in df.columns:
    # Extract section numbers as a separate feature
    df['section_number'] = df['Section'].str.extract(r'(\d+[A-Z]?)', expand=False)
    print(f"Extracted section numbers from {df['section_number'].notna().sum()} rows")

# Print some statistics
print("\nDataset Statistics:")
print(f"Total IPC sections: {len(df)}")
for col in df.columns:
    if df[col].dtype == object:
        non_empty = df[col].str.strip().ne('').sum()
        print(f"  - {col}: {non_empty}/{len(df)} non-empty values ({non_empty/len(df)*100:.1f}%)")

# Save preprocessed data
output_file = "ipc_sections_preprocessed.csv"
df.to_csv(output_file, index=False)
print(f"\nPreprocessed data saved to {output_file}")
print("You can now use this file for training your model.")

# Display a sample of the preprocessed data
print("\nSample of preprocessed data (first 2 rows, select columns):")
sample_cols = ['Section', 'Offense', 'cleaned_Offense', 'cleaned_combined_text']
print(df[sample_cols].head(2).to_string())