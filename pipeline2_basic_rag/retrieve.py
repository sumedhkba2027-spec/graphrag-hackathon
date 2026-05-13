import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chromadb")

collection = client.get_collection("graphrag_chunks")

def retrieve_chunks(query, top_k=3):

    # Convert query to embedding
    query_embedding = model.encode(query).tolist()

    # Search vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

if __name__ == "__main__":

    query = "What caused World War II?"

    results = retrieve_chunks(query)

    print("\nTop Retrieved Chunks:\n")

    for i, doc in enumerate(results["documents"][0]):

        print(f"--- Chunk {i+1} ---")
        print(doc[:1000])
        print("\n")