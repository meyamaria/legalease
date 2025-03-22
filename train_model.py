import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)
    df["text"] = df["Offense"] + " " + df["Description"]
    return df

def train_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    return vectorizer, tfidf_matrix

def get_best_match(query, df, vectorizer, tfidf_matrix, top_n=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][["Section", "Description", "Offense", "Punishment"]]
    return results

if __name__ == "__main__":
    file_path = "ipc_sections.csv"  # Ensure this is the correct path
    df = load_data(file_path)
    vectorizer, tfidf_matrix = train_tfidf(df)
    
    while True:
        query = input("Enter a keyword (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = get_best_match(query, df, vectorizer, tfidf_matrix)
        print("\nTop matching IPC sections:\n")
        print(results.to_string(index=False))
        print("\n" + "-"*50 + "\n")
