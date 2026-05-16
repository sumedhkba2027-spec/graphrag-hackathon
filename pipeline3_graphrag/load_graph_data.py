import pyTigerGraph as tg
import requests
import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("TG_HOST")
secret = os.getenv("TG_SECRET")
graph = os.getenv("TG_GRAPH")

# Get token via REST
url = f"{host}/gsql/v1/tokens"
response = requests.post(
    url,
    json={"secret": secret},
    headers={"Content-Type": "application/json"}
)

token = response.json()["token"]
print("✅ Token generated!")

# Connect with token
conn = tg.TigerGraphConnection(
    host=host,
    graphname=graph,
    gsPort="443",
    restppPort="443",
    tgCloud=True,
    apiToken=token
)

print("✅ Connected to TigerGraph!")
print("Vertex types:", conn.getVertexTypes())
print("Edge types:", conn.getEdgeTypes())