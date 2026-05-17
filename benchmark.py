import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline1_llm_only.pipeline1 import run_pipeline1
from pipeline2_basic_rag.pipeline2 import run_pipeline2
from pipeline3_graphrag.pipeline3 import run_pipeline3


questions = [
    "What did NEXON discuss about revenue and growth?",
    "What were the main challenges discussed in earning calls?",
    "What forward looking statements did companies make?",
    "What did Nissan discuss in their earning call?",
    "What financial results were discussed for fiscal year 2022?",
]


def summarize_result(name, result):
    return {
        "tokens": result["tokens_used"],
        "latency": result["latency_seconds"],
        "cost": result["cost_usd"],
        "answer": result["answer"],
    }


results = []

for question in questions:
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)

    print("\nRunning Pipeline 1...")
    r1 = run_pipeline1(question)

    print("\nRunning Pipeline 2...")
    r2 = run_pipeline2(question)

    print("\nRunning Pipeline 3...")
    r3 = run_pipeline3(question)

    results.append(
        {
            "question": question,
            "pipeline1": summarize_result("Pipeline 1", r1),
            "pipeline2": summarize_result("Pipeline 2", r2),
            "pipeline3": summarize_result("Pipeline 3", r3),
        }
    )

    print("\nRESULTS FOR THIS QUESTION:")
    print(f"{'Pipeline':<15} {'Tokens':>10} {'Latency':>10} {'Cost':>12}")
    print("-" * 50)
    print(f"{'P1 LLM Only':<15} {r1['tokens_used']:>10} {r1['latency_seconds']:>9}s {r1['cost_usd']:>12}")
    print(f"{'P2 Basic RAG':<15} {r2['tokens_used']:>10} {r2['latency_seconds']:>9}s {r2['cost_usd']:>12}")
    print(f"{'P3 GraphRAG':<15} {r3['tokens_used']:>10} {r3['latency_seconds']:>9}s {r3['cost_usd']:>12}")


with open("benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nResults saved to benchmark_results.json")

print(f"\n{'=' * 60}")
print("FINAL SUMMARY")
print("=" * 60)
avg_p1 = sum(r["pipeline1"]["tokens"] for r in results) / len(results)
avg_p2 = sum(r["pipeline2"]["tokens"] for r in results) / len(results)
avg_p3 = sum(r["pipeline3"]["tokens"] for r in results) / len(results)

reduction_vs_p1 = round((avg_p1 - avg_p3) / avg_p1 * 100, 1)
reduction_vs_p2 = round((avg_p2 - avg_p3) / avg_p2 * 100, 1)

print(f"Avg tokens - Pipeline 1: {avg_p1:.0f}")
print(f"Avg tokens - Pipeline 2: {avg_p2:.0f}")
print(f"Avg tokens - Pipeline 3: {avg_p3:.0f}")
print(f"\nToken reduction vs Pipeline 1: {reduction_vs_p1}%")
print(f"Token reduction vs Pipeline 2: {reduction_vs_p2}%")
