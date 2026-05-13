import json
import os

CHUNK_SIZE = 3000

with open("data/articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

chunks = []

chunk_id = 0

for article in articles:

    text = article["text"]

    for i in range(0, len(text), CHUNK_SIZE):

        chunk_text = text[i:i + CHUNK_SIZE]

        chunks.append({
            "chunk_id": chunk_id,
            "article_id": article["id"],
            "title": article["title"],
            "text": chunk_text
        })

        chunk_id += 1

os.makedirs("data/processed", exist_ok=True)

with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"\n✅ Created {len(chunks)} chunks")