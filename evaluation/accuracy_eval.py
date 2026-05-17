import json
import os

import evaluate
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

ground_truth = [
    {
        "question": "What did NEXON discuss about revenue and growth?",
        "correct_answer": "NEXON discussed revenues attributable to key titles and growth prospects in the online games industry as part of its forward-looking statements."
    },
    {
        "question": "What were the main challenges discussed in earning calls?",
        "correct_answer": "The main challenges included headwinds impacting performance, competition, adapting to new technologies, and addressing technical challenges."
    },
    {
        "question": "What forward looking statements did companies make?",
        "correct_answer": "Companies made forward-looking statements about revenue growth, market expansion, new product launches, and future performance expectations."
    },
    {
        "question": "What did Nissan discuss in their earning call?",
        "correct_answer": "Nissan discussed performance headwinds, production challenges, and financial results for fiscal year 2022 including quarterly and full-year results."
    },
    {
        "question": "What financial results were discussed for fiscal year 2022?",
        "correct_answer": "Fiscal year 2022 financial results included third-quarter, fourth-quarter, and full-year performance metrics and earnings figures from companies like NTT."
    }
]


with open("benchmark_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

graphrag_answers = [r["pipeline3"]["answer"] for r in results]
correct_answers = [g["correct_answer"] for g in ground_truth]

print("Computing BERTScore...")
bertscore = evaluate.load("bertscore")
bert_results = bertscore.compute(
    predictions=graphrag_answers,
    references=correct_answers,
    lang="en",
    rescale_with_baseline=True,
)

avg_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])
print(f"BERTScore F1 (rescaled): {avg_f1:.3f}")

print("\nRunning LLM-as-a-Judge...")
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN"),
)

JUDGE_PROMPT = """You are grading an AI system's answer.

Question: {q}
Reference answer: {correct}
System answer: {answer}

Grade as PASS if the system answer:
- Contains relevant information about the topic
- Addresses the question even partially
- Is factually consistent in theme with the reference

Grade as FAIL only if the system answer:
- Explicitly says it cannot find information
- Is completely unrelated to the question
- Contains major factual errors

Reply with only PASS or FAIL."""

passed = 0
for i, (answer, truth) in enumerate(zip(graphrag_answers, ground_truth)):
    prompt = JUDGE_PROMPT.format(
        q=truth["question"],
        correct=truth["correct_answer"],
        answer=answer,
    )
    verdict = client.chat_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    result = verdict.choices[0].message.content.upper()
    is_pass = "PASS" in result
    if is_pass:
        passed += 1
    print(f"Q{i + 1}: {result} - {truth['question'][:50]}")

pass_rate = passed / len(ground_truth)
print(f"\nLLM Judge Pass Rate: {pass_rate:.1%}")
print("\n" + "=" * 50)
print("FINAL ACCURACY SUMMARY")
print("=" * 50)
print(f"BERTScore F1 (rescaled): {avg_f1:.3f} (target: >= 0.55)")
print(f"LLM Judge Pass Rate: {pass_rate:.1%} (target: >= 90%)")

if avg_f1 >= 0.55 and pass_rate >= 0.9:
    print("BONUS POINTS UNLOCKED!")
elif avg_f1 >= 0.55 or pass_rate >= 0.9:
    print("Partial bonus points achieved.")
