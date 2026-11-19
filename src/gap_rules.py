import os
import json
import re
from collections import Counter

LEARNED_TRIGGERS_PATH = os.path.join("data", "learned_gap_triggers.json")

BASE_GAP_INDICATORS = [
    "however", "limitation", "limitations", "lack of", "future work",
    "unexplored", "further research", "requires further", "not addressed",
    "not investigated", "not evaluated", "remains unclear", "remains challenging",
    "open problem", "needs further study", "bottleneck", "unresolved",
    "inability to", "fails to generalize", "constrained by", "computational complexity",
    "poorly understood", "scarcity of", "challenging task", "drawback",
    "understudied", "gap in knowledge", "open question", "warrants further",
    "impediment", "inconclusive", "remains elusive", "pressing need"
]

GAP_INDICATORS = list(BASE_GAP_INDICATORS)

STOPWORDS = {
    "the", "a", "an", "in", "of", "to", "and", "is", "was", "for", "with",
    "on", "by", "as", "at", "from", "that", "this", "these", "those", "are",
    "were", "be", "been", "have", "has", "had", "it", "its", "we", "our",
    "they", "their", "which", "can", "could", "may", "might", "will", "would",
    "should", "not", "or", "but", "also", "into", "more", "such", "than", "between"
}

def load_learned_triggers() -> set:
    """Load persistent learned triggers merged with foundational base indicators."""
    triggers = set(BASE_GAP_INDICATORS)
    if os.path.exists(LEARNED_TRIGGERS_PATH):
        try:
            with open(LEARNED_TRIGGERS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    triggers.update(item.strip().lower() for item in data if item.strip())
        except Exception:
            pass
    return triggers

def save_learned_triggers(triggers: set | list) -> None:
    """Persist triggers to disk."""
    os.makedirs(os.path.dirname(LEARNED_TRIGGERS_PATH), exist_ok=True)
    clean_list = sorted(list(set(str(t).strip().lower() for t in triggers if str(t).strip())))
    with open(LEARNED_TRIGGERS_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_list, f, indent=2)

def induce_gap_vocabulary_from_corpus(sentences: list[str], top_n: int = 15) -> list[str]:
    """
    Self-learning routine: inspects cross-domain sentences (bio, AI, physics, chemistry),
    discovers contrastive, limitation, and boundary candidate expressions,
    and returns newly induced triggers, automatically persisting them.
    """
    if not sentences:
        return []

    context_patterns = [
        r"(?:remains|is still|leaves)\s+([a-z\-]+)",
        r"(?:lack of|scarcity of|paucity of)\s+([a-z\-]+)",
        r"(?:poorly|insufficiently|inadequately)\s+([a-z\-]+)",
        r"(?:challenge|bottleneck|drawback|obstacle)\s+in\s+([a-z\-]+)",
        r"(?:warrants|demands|necessitates)\s+further\s+([a-z\-]+)",
        r"(?:fails to|unable to|hard to)\s+([a-z\-]+)",
        r"(?:unresolved|understudied|unclear)\s+([a-z\-]+)"
    ]

    candidates = []

    for sent in sentences:
        s_lower = sent.lower()
        for pat in context_patterns:
            matches = re.findall(pat, s_lower)
            for m in matches:
                word = m.strip()
                if len(word) > 3 and word not in STOPWORDS:
                    candidates.append(word)

        if any(marker in s_lower for marker in ["however,", "nevertheless,", "despite", "although", "yet "]):
            words = re.findall(r"\b[a-z]{4,}\b", s_lower)
            for w in words:
                if w in ["unexplored", "challenging", "limited", "costly", "heuristic", "degradation", 
                         "bottleneck", "intractable", "disparity", "imbalance", "vulnerable", "scarce"]:
                    candidates.append(w)

    current_triggers = load_learned_triggers()
    counts = Counter(candidates)
    newly_discovered = []
    
    for word, freq in counts.most_common(top_n * 2):
        if word not in current_triggers and len(word) > 3:
            newly_discovered.append(word)
            current_triggers.add(word)
            if len(newly_discovered) >= top_n:
                break

    if newly_discovered:
        save_learned_triggers(current_triggers)

    return newly_discovered

def detect_gap_indicators(sentence: str) -> list[str]:
    """Match sentence against all active and dynamically learned gap triggers."""
    sent_lower = sentence.lower()
    active_triggers = load_learned_triggers()
    return [cue for cue in active_triggers if cue in sent_lower]

def rule_score(sentence: str, precomputed_matches: list[str] = None) -> float:
    """Calculate rule-based gap confidence score from matched triggers."""
    matches = precomputed_matches if precomputed_matches is not None else detect_gap_indicators(sentence)
    return min(len(matches) / 2.0, 1.0) if matches else 0.0