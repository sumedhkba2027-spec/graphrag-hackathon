import json
import tiktoken

with open("data/articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

enc = tiktoken.get_encoding("cl100k_base")

total_tokens = 0

for article in articles:
    tokens = len(enc.encode(article["text"]))
    total_tokens += tokens

print(f"\nTotal Articles: {len(articles)}")
print(f"Total Tokens: {total_tokens:,}")
print(f"Average Tokens per Article: {total_tokens // len(articles):,}")