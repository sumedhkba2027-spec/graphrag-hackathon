import time
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def run_pipeline1(question):
    start_time = time.time()
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question
    )
    
    end_time = time.time()
    
    return {
        "pipeline": "Pipeline 1 - LLM Only",
        "question": question,
        "retrieved_chunks": 0,
        "answer": response.text,
        "tokens_used": response.usage_metadata.total_token_count,
        "latency_seconds": round(end_time - start_time, 2),
        "cost_usd": round(response.usage_metadata.total_token_count * 0.000001, 6)
    }

# Test it
if __name__ == "__main__":
    question = "What were the main causes of World War II?"
    result = run_pipeline1(question)
    
    print(f"Question: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\n--- Metrics ---")
    print(f"Tokens used: {result['tokens_used']}")
    print(f"Latency: {result['latency_seconds']}s")
    print(f"Cost: ${result['cost_usd']}")