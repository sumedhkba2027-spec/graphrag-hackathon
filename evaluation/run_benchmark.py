import sys
import os

sys.path.append(os.path.abspath("."))
import json

from pipeline1_llm_only.pipeline1 import run_pipeline1
from pipeline2_basic_rag.pipeline2 import run_pipeline2

# Load questions
with open("evaluation/questions.json", "r") as f:
    questions = json.load(f)

results = []

for item in questions:

    question = item["question"]

    print(f"\nRunning Question:")
    print(question)

    # Pipeline 1
    p1 = run_pipeline1(question)

    # Pipeline 2
    p2 = run_pipeline2(question)

    results.append({
        "question": question,
        "pipeline1": p1,
        "pipeline2": p2
    })

# Save benchmark results
with open("evaluation/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ Benchmark completed")