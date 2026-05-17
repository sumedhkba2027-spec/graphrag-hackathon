import json
import os
import spacy
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

nlp = spacy.load("en_core_web_sm")

HOST = os.getenv("TG_HOST")
SECRET = os.getenv("TG_SECRET")
GRAPH = os.getenv("TG_GRAPH")

conn = tg.TigerGraphConnection(
    host=HOST, graphname=GRAPH, gsqlSecret=SECRET
)
token = conn.getToken(SECRET)
if isinstance(token, tuple):
    token = token[0]
conn = tg.TigerGraphConnection(
    host=HOST, graphname=GRAPH, gsqlSecret=SECRET, apiToken=token
)
print("✅ Connected to TigerGraph!")

with open("data/articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)
print(f"✅ Loaded {len(articles)} articles")

for article in articles:
    doc_id = str(article["id"])
    title = article["title"]
    text = article["text"][:5000]

    # Insert Document — using actual attribute names from schema
    conn.upsertVertex("Document", doc_id, {
        "title": title,
        "text": text,
        "content": text,  # schema has both text and content
        "name": title
    })
    print(f"📄 Added document: {title}")

    # Extract entities
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        entity = ent.text.strip()
        if len(entity) > 2:
            entities.add(entity)

    # Add entity vertices and MENTIONS edges
    for entity in entities:
        # Entity PRIMARY_ID is id, name is attribute
        entity_id = entity.replace(" ", "_").lower()
        conn.upsertVertex("Entity", entity_id, {
            "name": entity,
            "title": entity,
            "content": entity,
            "source": title
        })
        conn.upsertEdge("Entity", entity_id, "Mentions", "Document", doc_id)

    # Create RELATED_TO edges between entities
    entity_list = list(entities)
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            id_i = entity_list[i].replace(" ", "_").lower()
            id_j = entity_list[j].replace(" ", "_").lower()
            conn.upsertEdge("Entity", id_i, "RELATED_TO", "Entity", id_j)

    print(f"🔗 Added {len(entities)} entities")

print("\n✅ Graph data loading complete!")
print("Total Documents:", conn.getVertexCount("Document"))
print("Total Entities:", conn.getVertexCount("Entity"))