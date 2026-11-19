"""
Dynamic Gap Learner Module
Autonomous vocabulary induction engine:
Scans cross-domain research literature for contrastive, limitation, and boundary patterns,
inducting newly discovered indicator keywords into memory.
"""

import os
import json
import re
from collections import Counter
from src.gap_rules import (
    load_learned_triggers,
    save_learned_triggers,
    detect_gap_indicators,
    rule_score,
    BASE_GAP_INDICATORS,
    LEARNED_TRIGGERS_PATH
)

CONTEXT_PATTERNS = [
    r"(?:remains|is still|leaves)\s+([a-z\-]+)",
    r"(?:lack of|scarcity of|paucity of)\s+([a-z\-]+)",
    r"(?:poorly|insufficiently|inadequately)\s+([a-z\-]+)",
    r"(?:challenge|bottleneck|drawback|obstacle)\s+in\s+([a-z\-]+)",
    r"(?:warrants|demands|necessitates)\s+further\s+([a-z\-]+)",
    r"(?:fails to|unable to|hard to)\s+([a-z\-]+)",
    r"(?:unresolved|understudied|unclear)\s+([a-z\-]+)"
]

GENERAL_CONTRAST_WORDS = [
    "unexplored", "challenging", "limited", "costly", "heuristic",
    "degradation", "bottleneck", "intractable", "disparity", "imbalance",
    "vulnerable", "scarce", "prohibitive", "incomputable", "cytotoxic",
    "unviable", "non-trivial", "unfeasible", "degrade", "bottlenecks"
]

def induce_gap_vocabulary_from_corpus(sentences: list[str], top_n: int = 15) -> list[str]:
    """
    Extracts candidate limitation trigger words from newly ingested academic sentences.
    Persists valid candidate discoveries into the global learned triggers store.
    """
    if not sentences:
        return []

    from src.gap_rules import STOPWORDS

    candidates = []
    for sent in sentences:
        s_lower = sent.lower()
        for pat in CONTEXT_PATTERNS:
            for match in re.findall(pat, s_lower):
                word = match.strip()
                if len(word) > 3 and word not in STOPWORDS:
                    candidates.append(word)

        if any(marker in s_lower for marker in ["however,", "nevertheless,", "despite", "although", "yet "]):
            words = re.findall(r"\b[a-z]{4,}\b", s_lower)
            for w in words:
                if w in GENERAL_CONTRAST_WORDS:
                    candidates.append(w)

    current_triggers = load_learned_triggers()
    counts = Counter(candidates)
    newly_discovered = []

    for word, _ in counts.most_common(top_n * 2):
        if word not in current_triggers and len(word) > 3:
            newly_discovered.append(word)
            current_triggers.add(word)
            if len(newly_discovered) >= top_n:
                break

    if newly_discovered:
        save_learned_triggers(current_triggers)

    return newly_discovered

def get_lexicon_summary() -> dict:
    """Returns statistics regarding active base vs learned triggers."""
    active = load_learned_triggers()
    base_set = set(BASE_GAP_INDICATORS)
    learned_only = active - base_set
    return {
        "total_active_triggers": len(active),
        "base_triggers": len(base_set),
        "dynamically_learned_triggers": len(learned_only),
        "learned_keywords": sorted(list(learned_only))
    }
