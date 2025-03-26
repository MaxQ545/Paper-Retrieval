import json
import re
import argparse
import torch
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import os
import pickle


# Data cleaning function
def clean_text(text):
    """Clean the text by removing HTML tags and special characters"""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()


# Load data
def load_data(file_path):
    titles_and_abstracts = []
    if os.path.exists("titles_and_abstracts.pkl"):
        with open("titles_and_abstracts.pkl", "rb") as f:
            titles_and_abstracts = pickle.load(f)
            return titles_and_abstracts

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            try:
                data = json.loads(line)
                title = data.get('title', '').strip()
                abstract = data.get('abstract', '').strip()
                if title and abstract:
                    cleaned_abstract = clean_text(abstract)
                    titles_and_abstracts.append((title, cleaned_abstract))
            except json.JSONDecodeError:
                continue
    print(f"Total {len(titles_and_abstracts)} titles and abstracts extracted.")
    pickle.dump(titles_and_abstracts, open("titles_and_abstracts.pkl", "wb"))
    return titles_and_abstracts


# Train and generate embeddings
def train_embeddings(titles_and_abstracts, model_path='saved_model', index_path='arxiv_titles_abstracts.index', batch_size=1024):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load or initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Generate embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(titles_and_abstracts), batch_size), desc="Generating embeddings"):
        batch = [abstract for _, abstract in titles_and_abstracts[i:i + batch_size]]
        batch_embeddings = model.encode(
            batch,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Embedding generation complete, shape: {embeddings.shape}")

    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Create and save the index
    embeddings_np = embeddings.cpu().numpy()
    dimension = embeddings_np.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings_np)
    faiss.write_index(index, index_path)
    print(f"Index construction complete and saved to {index_path}, containing {index.ntotal} vectors.")

    return model, index


# Search function
def search(query, model, index, titles_and_abstracts, k=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query_embedding = model.encode(
        [query],
        convert_to_tensor=True,
        device=device
    ).cpu().numpy()

    distances, indices = index.search(query_embedding, k)
    results = [(titles_and_abstracts[idx][0], titles_and_abstracts[idx][1], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results


# Main function
def main():
    parser = argparse.ArgumentParser(description="Semantic Search System: Train or Search")
    parser.add_argument('--mode', type=str, choices=['train', 'search'], required=True, help="Mode: 'train' or 'search'")
    parser.add_argument('--data', type=str, default='arxiv-metadata.jsonl', help="Data file path")
    parser.add_argument('--model_path', type=str, default='saved_model', help="Model save path")
    parser.add_argument('--index_path', type=str, default='arxiv_titles_abstracts.index', help="Index save path")
    parser.add_argument('--query', type=str, help="Query sentence for search mode")
    args = parser.parse_args()

    if args.mode == 'train':
        # Training mode
        titles_and_abstracts = load_data(args.data)
        train_embeddings(titles_and_abstracts, args.model_path, args.index_path)

    elif args.mode == 'search':
        # Search mode
        if not os.path.exists(args.model_path) or not os.path.exists(args.index_path):
            print("Error: Model or index file does not exist. Please run the training mode first.")
            return

        # Load data, model, and index
        titles_and_abstracts = load_data(args.data)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(args.model_path, device=device)
        index = faiss.read_index(args.index_path)
        print(f"Model and index loaded, index contains {index.ntotal} vectors.")

        # Execute search
        if not args.query:
            print("Error: Query is required in search mode (--query)")
            return
        results = search(args.query, model, index, titles_and_abstracts, k=10)
        print("Search Results:")
        for i, (title, abstract, distance) in enumerate(results, 1):
            print(f"{i}. Distance: {distance:.4f}\nTitle: {title}\nAbstract: {abstract[:200]}...\n")


if __name__ == "__main__":
    main()
