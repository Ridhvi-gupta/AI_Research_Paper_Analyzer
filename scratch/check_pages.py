import sys, os
sys.path.insert(0, os.path.abspath('.'))

# We can inspect if any module or logic in app.py fails for each page name
PAGES = [
    "🏠 Home",
    "🚀 Upload & Analyze Hub",
    "📄 Multi-Format Papers Library",
    "📈 Text Analysis & EDA",
    "🔗 Cosine Similarity",
    "🧩 Interdisciplinary Clustering",
    "🔍 Adaptive Gap Mining & Suggestions",
    "🧠 Learned Vocabulary & AI Rules",
    "📑 Dossier Reports",
    "🔬 Pipeline Architecture",
    "❓ FAQ & Research Q&A",
    "⚙ Settings & Platform"
]

print("Available pages count:", len(PAGES))
for p in PAGES:
    print("Page:", p)
