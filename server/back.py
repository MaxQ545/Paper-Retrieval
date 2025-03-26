from flask import Flask, request, jsonify
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and index (they are pre-trained and saved)
model_path = '../saved_model'
index_path = '../arxiv_titles_abstracts.index'
titles_and_abstracts = pickle.load(open("../titles_and_abstracts.pkl", "rb"))

# Load model and FAISS index
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(model_path, device=device)
index = faiss.read_index(index_path)


# Search endpoint
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])

    # Generate query embedding
    query_embedding = model.encode([query], convert_to_tensor=True, device=device).cpu().numpy()

    # Perform the search
    k = 10  # Return top 10 results
    distances, indices = index.search(query_embedding, k)

    results = []
    for i, idx in enumerate(indices[0]):
        title, abstract = titles_and_abstracts[idx]
        results.append({
            'title': title,
            'abstract': abstract,
            'distance': float(distances[0][i])
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host='172.16.0.0')
