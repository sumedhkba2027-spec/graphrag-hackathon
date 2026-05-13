import time
import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

# Gemini client
client_gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ChromaDB
client_chroma = chromadb.PersistentClient(
    path="chromadb"
)

collection = client_chroma.get_collection(
    "graphrag_chunks"
)

def retrieve_chunks(query, top_k=3):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]

def run_pipeline2(question):

    start_time = time.time()

    # Retrieve context
    retrieved_chunks = retrieve_chunks(question)

    context = "\n\n".join(retrieved_chunks)

    # Build prompt
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
"""

    # Gemini response
    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    end_time = time.time()

    return {
        "pipeline": "Pipeline 2 - Basic RAG",
        "question": question,
        "answer": response.text,
        "retrieved_chunks": len(retrieved_chunks),
        "tokens_used": response.usage_metadata.total_token_count,
        "latency_seconds": round(end_time - start_time, 2),
        "cost_usd": round(
            response.usage_metadata.total_token_count * 0.000001,
            6
        )
    }

if __name__ == "__main__":

    question = "What caused World War II?"

    result = run_pipeline2(question)

    print(f"\nQuestion: {result['question']}")

    print(f"\nAnswer:\n{result['answer']}")

    print(f"\n--- Metrics ---")
    print(f"Retrieved Chunks: {result['retrieved_chunks']}")
    print(f"Tokens used: {result['tokens_used']}")
    print(f"Latency: {result['latency_seconds']}s")
    print(f"Cost: ${result['cost_usd']}")