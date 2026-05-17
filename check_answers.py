import json

with open("benchmark_results.json") as f:
    results = json.load(f)

for r in results:
    print(f"\nQ: {r['question']}")
    print(f"P3 Answer: {r['pipeline3']['answer'][:200]}")
    print("-"*50)