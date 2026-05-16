import pyTigerGraph as tg
import os
from dotenv import load_dotenv

load_dotenv()

conn = tg.TigerGraphConnection(
    host=os.getenv("TG_HOST"),
    username=os.getenv("TG_USERNAME"),
    password=os.getenv("TG_PASSWORD")
)

print("\nConnected Successfully!")

print("\nTigerGraph Version:")
print(conn.getVersion())