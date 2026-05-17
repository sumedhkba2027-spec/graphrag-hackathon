import os
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

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

print(conn.getEdgeType("Mentions"))
print(conn.getEdgeType("RELATED_TO"))