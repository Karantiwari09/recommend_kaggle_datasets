import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

def train_model():
    df = pd.read_csv("data/kaggle-preprocessed.csv")

    df['tags'] = (
        df['Dataset_name'].fillna('') + ' ' +
        df['Type_of_file'].fillna('') + ' ' +
        df['Medals'].fillna('') + ' ' +
        df['size'].fillna('')
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["tags"])

    model = NearestNeighbors(n_neighbors=10, metric='cosine')
    model.fit(tfidf_matrix)

    os.makedirs("model", exist_ok=True)
    joblib.dump((model, vectorizer, df), "model/recommender_model.pkl")
    print("✅ Model saved as model/recommender_model.pkl")

if __name__ == "__main__":
    train_model()
