import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv()

conn = tg.TigerGraphConnection(
    host=os.getenv("TG_HOST"),
    graphname="GraphRAG",
    gsPort="443",
    restppPort="443",
    tgCloud=True
)

token = conn.getToken(os.getenv("TG_SECRET"))[0]
print("✅ Token generated!")

conn = tg.TigerGraphConnection(
    host=os.getenv("TG_HOST"),
    graphname="GraphRAG",
    gsPort="443",
    restppPort="443",
    tgCloud=True,
    apiToken=token
)

print("✅ Connected to GraphRAG graph!")
print("Vertex types:", conn.getVertexTypes())
print("Edge types:", conn.getEdgeTypes())