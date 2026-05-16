import os
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("TG_HOST")
SECRET = os.getenv("TG_SECRET")
GRAPH = os.getenv("TG_GRAPH")

conn = tg.TigerGraphConnection(
    host=HOST,
    graphname=GRAPH,
    gsqlSecret=SECRET,
)
token = conn.getToken(SECRET)
if isinstance(token, tuple):
    token = token[0]

conn = tg.TigerGraphConnection(
    host=HOST,
    graphname=GRAPH,
    gsqlSecret=SECRET,
    apiToken=token,
)

result = conn.gsql(f"""
USE GRAPH {GRAPH}
ALTER GRAPH {GRAPH} (
    ADD VERTEX Document,
    ADD VERTEX Entity,
    ADD EDGE MENTIONS,
    ADD EDGE RELATED_TO
)
""")
print("Result:", result)