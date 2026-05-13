from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API Key Loaded:", api_key is not None)

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence."
)

print("\nModel Response:")
print(response.text)
print("Tokens used:", response.usage_metadata.total_token_count)