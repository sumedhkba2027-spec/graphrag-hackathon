import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.PersistentClient(path="chromadb")

# Create collection
collection = client.get_or_create_collection(
    name="graphrag_chunks"
)

# Load chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# Process chunks
for chunk in chunks:

    embedding = model.encode(chunk["text"]).tolist()

    collection.add(
        ids=[str(chunk["chunk_id"])],
        embeddings=[embedding],
        documents=[chunk["text"]],
        metadatas=[{
            "title": chunk["title"]
        }]
    )

print("\n✅ Vector database created successfully")