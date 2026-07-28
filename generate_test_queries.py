import json
import random
import time
from src.llm_engine import LLMEngine
from concurrent.futures import ThreadPoolExecutor

llm = LLMEngine()

with open("data/scifact_corpus_full.json") as f:
    corpus = json.load(f)

# Get 200 random documents to ensure we get >100 queries after some failures
docs = random.sample(list(corpus.values()), 200)
queries = []

def generate_query(doc):
    prompt = f"Given this text snippet, write ONE short, specific question that can be answered by the text. Return ONLY the question, nothing else.\n\nText: {doc[:500]}"
    try:
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        return None

print("Generating up to 200 queries (this may take a minute)...")
with ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(generate_query, docs)

for q in results:
    if q and len(q) > 10 and "?" in q:
        queries.append(q)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SciFact Test Queries</title>
    <style>
        :root {{
            --bg: #F9FAFB;
            --surface: #FFFFFF;
            --border: #E5E7EB;
            --text: #111827;
            --text-m: #374151;
            --accent: #000000;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
        }}
        h1 {{
            font-size: 24px;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        p.subtitle {{
            color: var(--text-m);
            margin-bottom: 30px;
            font-size: 15px;
        }}
        .query-list {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            list-style-type: decimal;
        }}
        li {{
            margin-bottom: 12px;
            padding-left: 8px;
            font-size: 14.5px;
            color: var(--text-m);
        }}
        li:last-child {{
            margin-bottom: 0;
        }}
    </style>
</head>
<body>
    <h1>SciFact Test Queries</h1>
    <p class="subtitle">Here are some example {len(queries)} questions generated from the corpus that the Vector-ARC system can answer.</p>
    <ol class="query-list">
"""

for q in queries:
    html_content += f"        <li>{q}</li>\n"

html_content += """    </ol>
</body>
</html>"""

with open("demo/test_queries.html", "w") as f:
    f.write(html_content)

print(f"Saved {len(queries)} queries to demo/test_queries.html")
