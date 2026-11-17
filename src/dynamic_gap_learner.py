import os
import json
import re
from gensim.models import Word2Vec

LEARNED_TRIGGERS_FILE = os.path.join("data", "learned_gap_triggers.json")

# Base Conceptual Anchors for Research Gaps
BASE_ANCHORS = [
    "limitation", "drawback", "bottleneck", "challenge", "unresolved",
    "constraint", "degrade", "insufficiency", "failure", "scarcity",
    "prohibitive", "unviable", "lack", "underrepresented", "elusive"
]

def load_learned_triggers() -> set:
    """Loads discovered & validated gap indicators from persistent JSON."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(LEARNED_TRIGGERS_FILE):
        try:
            with open(LEARNED_TRIGGERS_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    # Initialize with default seed set
    default_set = set(BASE_ANCHORS + [
        "however", "future work", "remains", "persist", "fails to", "fails when", "memory footprint", "imbalance"
    ])
    save_learned_triggers(default_set)
    return default_set

def save_learned_triggers(triggers: set):
    """Saves discovered gap vocabulary to disk."""
    os.makedirs("data", exist_ok=True)
    with open(LEARNED_TRIGGERS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(triggers)), f, indent=2)

def induce_gap_vocabulary_from_corpus(sentences: list[str]) -> list[str]:
    """
    Continuous Self-Learning Engine:
    Trains Word2Vec on the user's uploaded papers and scans the embedding space
    to discover novel, domain-specific gap indicator words.
    """
    tokenized_corpus = []
    for s in sentences:
        tokens = re.findall(r'\b[A-Za-z]{3,}\b', s.lower())
        if tokens:
            tokenized_corpus.append(tokens)

    if len(tokenized_corpus) < 5:
        return []

    try:
        w2v = Word2Vec(tokenized_corpus, vector_size=80, window=4, min_count=1, sg=1, epochs=10)
    except Exception:
        return []

    existing_triggers = load_learned_triggers()
    newly_discovered = set()

    for anchor in BASE_ANCHORS:
        if anchor in w2v.wv:
            try:
                neighbors = w2v.wv.most_similar(anchor, topn=4)
                for word, sim in neighbors:
                    if sim >= 0.40 and len(word) > 3 and word not in existing_triggers:
                        newly_discovered.add(word)
            except Exception:
                continue

    if newly_discovered:
        updated = existing_triggers.union(newly_discovered)
        save_learned_triggers(updated)

    return sorted(list(newly_discovered))