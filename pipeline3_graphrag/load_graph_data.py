import json
import os
import spacy
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

HOST = os.getenv("TG_HOST")
SECRET = os.getenv("TG_SECRET")
GRAPH = os.getenv("TG_GRAPH")  # "GraphRAG"

# Devanshu's fix — pass secret in FIRST connection
conn = tg.TigerGraphConnection(
    host=HOST,
    graphname=GRAPH,
    gsqlSecret=SECRET,
)

token = conn.getToken(SECRET)
if isinstance(token, tuple):
    token = token[0]

# Reconnect with both secret AND token
conn = tg.TigerGraphConnection(
    host=HOST,
    graphname=GRAPH,
    gsqlSecret=SECRET,
    apiToken=token,
)
print("✅ Connected to TigerGraph!")

# Load articles
with open("data/articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

print(f"✅ Loaded {len(articles)} articles")

# Process articles
for article in articles:
    doc_id = str(article["id"])
    title = article["title"]
    text = article["text"][:5000]

    # Insert Document vertex
    conn.upsertVertex("Document", doc_id, {
        "title": title,
        "text": text
    })
    print(f"📄 Added document: {title}")

    # Extract entities using spaCy
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        entity = ent.text.strip()
        if len(entity) > 2:
            entities.add(entity)

    # Add entity vertices and MENTIONS edges
    for entity in entities:
        conn.upsertVertex("Entity", entity)
        conn.upsertEdge("Document", doc_id, "MENTIONS", "Entity", entity)

    # Create RELATED_TO edges between entities
    entity_list = list(entities)
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            conn.upsertEdge("Entity", entity_list[i], "RELATED_TO", "Entity", entity_list[j])

    print(f"🔗 Added {len(entities)} entities")

print("\n✅ Graph data loading complete!")
print("Total Documents:", conn.getVertexCount("Document"))
print("Total Entities:", conn.getVertexCount("Entity"))