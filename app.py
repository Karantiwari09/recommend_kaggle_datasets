from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model, vectorizer, dataset = joblib.load("model/recommender_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.form["interest"]

    input_vec = vectorizer.transform([user_input])

    distances, indices = model.kneighbors(input_vec, n_neighbors=20) 

    matched_data = dataset.iloc[indices[0]][[
        "Dataset_name", "Author_name", "Type_of_file", "Medals", "size", "Dataset_link"
    ]].reset_index(drop=True)

    return render_template("result.html", interest=user_input, recommendations=matched_data.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
