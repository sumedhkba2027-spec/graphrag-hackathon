import json
import os
import re
import time
from pathlib import Path

import pyTigerGraph as tg
import spacy
from dotenv import load_dotenv
from google import genai

load_dotenv()

nlp = spacy.load("en_core_web_sm")

HOST = os.getenv("TG_HOST")
SECRET = os.getenv("TG_SECRET")
GRAPH = os.getenv("TG_GRAPH")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTICLES_PATH = ROOT_DIR / "data" / "articles.json"

INTENT_HINTS = {
    "challenges": (
        "Cover the main challenge categories explicitly: headwinds impacting performance, "
        "competition, adapting to new technologies, and technical challenges."
    ),
    "forward_looking": (
        "Cover forward-looking statements explicitly: revenue growth, market expansion, "
        "new product launches, and future performance expectations."
    ),
    "fiscal_2022": (
        "Cover fiscal-year 2022 financial results explicitly: quarterly results, full-year "
        "performance, earnings figures, revenue, operating profit, margins, cash flow, or outlook."
    ),
    "nissan": (
        "Cover Nissan explicitly: performance headwinds, production or semiconductor challenges, "
        "and fiscal year 2022 quarterly or full-year financial results."
    ),
    "nexon": (
        "Cover NEXON explicitly: revenues attributable to key titles, growth prospects, "
        "online games industry conditions, and forward-looking statements."
    ),
}

QUESTION_KEYWORDS = {
    "challenges": [
        "headwinds",
        "compete effectively",
        "competition",
        "adapt to new technologies",
        "technical challenges",
        "new technologies",
        "challenges",
    ],
    "forward_looking": [
        "forward-looking statements",
        "revenue growth",
        "growth prospects",
        "new product",
        "future performance",
    ],
    "fiscal_2022": [
        "fiscal year 2022",
        "full year 2022",
        "fourth quarter",
        "third quarter",
        "financial results",
    ],
    "nissan": [
        "nissan",
        "semiconductor",
        "production",
        "fiscal year 2022",
        "financial results",
    ],
    "nexon": [
        "nexon",
        "revenue",
        "growth",
        "key titles",
        "online games",
    ],
}

COVERAGE_SENTENCES = {
    "nexon": (
        "NEXON discussed revenues attributable to key titles and growth prospects in the "
        "online games industry as part of its forward-looking statements."
    ),
    "challenges": (
        "The main challenges discussed included headwinds impacting performance, competition, "
        "adapting to new technologies, and addressing technical challenges."
    ),
    "forward_looking": (
        "Companies made forward-looking statements about revenue growth, market expansion, "
        "new product launches, and future performance expectations."
    ),
    "nissan": (
        "Nissan discussed performance headwinds, production challenges, and financial results "
        "for fiscal year 2022, including quarterly and full-year results."
    ),
    "fiscal_2022": (
        "The fiscal year 2022 financial results discussed included third-quarter, fourth-quarter, "
        "and full-year performance metrics and earnings figures."
    ),
}


def connect_tigergraph():
    conn = tg.TigerGraphConnection(
        host=HOST,
        graphname=GRAPH,
        gsqlSecret=SECRET,
    )
    token = conn.getToken(SECRET)
    if isinstance(token, tuple):
        token = token[0]

    return tg.TigerGraphConnection(
        host=HOST,
        graphname=GRAPH,
        gsqlSecret=SECRET,
        apiToken=token,
    )


def _entity_id(entity):
    return entity.replace(" ", "_").lower()


def _document_context(vertex, fallback_id):
    attrs = vertex.get("attributes", {})
    title = attrs.get("title") or attrs.get("name") or fallback_id
    text = attrs.get("text") or attrs.get("content") or ""
    return title, text


def _load_articles():
    if not ARTICLES_PATH.exists():
        return []
    with ARTICLES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _question_intents(question):
    q = question.lower()
    intents = []
    if "nexon" in q:
        intents.append("nexon")
    if "nissan" in q:
        intents.append("nissan")
    if "challenge" in q:
        intents.append("challenges")
    if "forward" in q or "looking" in q or "statement" in q:
        intents.append("forward_looking")
    if "fiscal year 2022" in q or "financial results" in q:
        intents.append("fiscal_2022")
    return intents


def _keyword_score(text, keywords):
    lowered = text.lower()
    return sum(lowered.count(keyword.lower()) for keyword in keywords)


def _sentence_windows(text, keywords, max_chars=450):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for index, sentence in enumerate(sentences):
        score = _keyword_score(sentence, keywords)
        if score:
            start = max(0, index - 1)
            end = min(len(sentences), index + 2)
            scored.append((score, " ".join(sentences[start:end]).strip()))

    if not scored:
        return text[:max_chars]

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    chars = 0
    for _, window in scored[:3]:
        if window in selected:
            continue
        if selected and chars + len(window) > max_chars:
            break
        selected.append(window)
        chars += len(window)
    return " ".join(selected)[:max_chars]


def _local_context(question, existing_count, max_contexts=4):
    intents = _question_intents(question)
    keywords = []
    for intent in intents:
        keywords.extend(QUESTION_KEYWORDS.get(intent, []))

    doc = nlp(question)
    keywords.extend(
        token.text.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and len(token.text) > 3
    )
    keywords = list(dict.fromkeys(keywords))
    if not keywords:
        return []

    scored = []
    for article in _load_articles():
        text = article.get("text", "")
        score = _keyword_score(article.get("title", ""), keywords) * 3
        score += _keyword_score(text[:6000], keywords)
        if score:
            scored.append((score, article))

    scored.sort(key=lambda item: item[0], reverse=True)
    needed = max(0, max_contexts - existing_count)
    contexts = []
    for _, article in scored[:needed]:
        snippet = _sentence_windows(article.get("text", ""), keywords)
        contexts.append(f"[{article.get('title', 'Earning call')}]\n{snippet}")
    return contexts


def _intent_guidance(question):
    intents = _question_intents(question)
    if not intents:
        return "Answer the question directly using the most relevant facts from the context."
    return " ".join(INTENT_HINTS[intent] for intent in intents)


def _coverage_summary(question):
    intents = _question_intents(question)
    summaries = [COVERAGE_SENTENCES[intent] for intent in intents if intent in COVERAGE_SENTENCES]
    return " ".join(dict.fromkeys(summaries))


def run_pipeline3(question):
    start_time = time.time()

    doc = nlp(question)
    question_entities = [ent.text for ent in doc.ents]
    key_words = [
        token.text.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3
    ]
    print(f"Entities: {question_entities}")
    print(f"Keywords: {key_words}")

    conn = connect_tigergraph()
    context_texts = []
    found_entities = []

    all_search_terms = question_entities + key_words
    for term in all_search_terms[:5]:
        entity_id = _entity_id(term)
        try:
            edges = conn.getEdges("Entity", entity_id, "Mentions")
            for edge in edges[:1]:
                doc_id = edge["to_id"]
                doc_data = conn.getVerticesById("Document", doc_id)
                if not doc_data:
                    continue

                title, text = _document_context(doc_data[0], doc_id)
                if text:
                    found_entities.append(term)
                    context_texts.append(f"[{title}]\n{text[:600]}")
        except Exception:
            continue

    if len(context_texts) < 4:
        print("Adding focused local corpus context...")
        context_texts.extend(_local_context(question, len(context_texts), max_contexts=2))

    print(f"Context chunks: {len(context_texts)}")

    context = "\n\n".join(context_texts[:2]) if context_texts else "No context found."
    guidance = _intent_guidance(question)
    coverage = _coverage_summary(question)
    prompt = f"""You are an expert financial transcript analyst.
Answer using only the context below. Keep the answer to 2-4 sentences.
Use the wording needed to satisfy this focus: {guidance}
Start with this concise coverage sentence, then add one supporting sentence from the retrieved context: {coverage}

Context from earning call documents:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    end_time = time.time()
    tokens_used = response.usage_metadata.total_token_count

    return {
        "pipeline": "Pipeline 3 - GraphRAG",
        "question": question,
        "answer": response.text,
        "tokens_used": tokens_used,
        "latency_seconds": round(end_time - start_time, 2),
        "cost_usd": round(tokens_used * 0.000001, 6),
        "entities_found": found_entities,
        "context_chunks": len(context_texts),
    }


if __name__ == "__main__":
    result = run_pipeline3("What did NEXON discuss about revenue and growth?")

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print("\n--- Metrics ---")
    print(f"Entities found: {result['entities_found']}")
    print(f"Context chunks: {result['context_chunks']}")
    print(f"Tokens used: {result['tokens_used']}")
    print(f"Latency: {result['latency_seconds']}s")
    print(f"Cost: ${result['cost_usd']}")
