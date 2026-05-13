import wikipedia
import json
import os

topics = [
    "Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin",
    "World War II", "World War I", "French Revolution", "Roman Empire",
    "DNA", "Black hole", "Quantum mechanics", "Theory of relativity",
    "Renaissance", "Industrial Revolution", "Cold War", "Space Race",
    "Artificial intelligence", "Internet", "Climate change", "Evolution",
    "William Shakespeare", "Leonardo da Vinci", "Napoleon Bonaparte",
    "Abraham Lincoln", "Mahatma Gandhi", "Nelson Mandela", "Cleopatra",
    "Alexander the Great", "Julius Caesar", "Christopher Columbus",
    "Amazon rainforest", "Great Wall of China", "Eiffel Tower",
    "United Nations", "European Union", "NASA", "CERN",
    "Photosynthesis", "Gravity", "Electricity", "Vaccination",
    "Penicillin", "Human genome", "Big Bang", "Solar system",
    "Python programming", "Machine learning", "Neural network",
    "Blockchain", "Cryptocurrency", "Electric vehicle", "Nuclear energy"
]

os.makedirs("data", exist_ok=True)
articles = []
failed = []

for i, topic in enumerate(topics):
    try:
        page = wikipedia.page(topic)
        articles.append({
            "id": i,
            "title": topic,
            "text": page.content
        })
        print(f"✅ {i+1}/{len(topics)}: {topic}")
    except Exception as e:
        failed.append(topic)
        print(f"❌ Failed: {topic} -> {e}")

with open("data/articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved {len(articles)} articles")
print(f"❌ Failed: {len(failed)}")  