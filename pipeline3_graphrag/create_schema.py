import os
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("TG_HOST")
SECRET = os.getenv("TG_SECRET")
GRAPH = os.getenv("TG_GRAPH")

# Connect
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

print("✅ Connected!")

# Create schema
result = conn.gsql(f"""
USE GLOBAL

CREATE VERTEX Document (
    PRIMARY_ID id STRING,
    title STRING,
    text STRING
) WITH primary_id_as_attribute="true"

CREATE VERTEX Entity (
    PRIMARY_ID name STRING
) WITH primary_id_as_attribute="true"

CREATE UNDIRECTED EDGE MENTIONS (
    FROM Document,
    TO Entity
)

CREATE UNDIRECTED EDGE RELATED_TO (
    FROM Entity,
    TO Entity
)

USE GRAPH {GRAPH}

ADD VERTEX Document TO GRAPH {GRAPH}
ADD VERTEX Entity TO GRAPH {GRAPH}
ADD EDGE MENTIONS TO GRAPH {GRAPH}
ADD EDGE RELATED_TO TO GRAPH {GRAPH}
""")

print("Schema result:", result)
print("✅ Done!")