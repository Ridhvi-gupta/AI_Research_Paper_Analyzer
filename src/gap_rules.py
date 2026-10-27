GAP_INDICATORS = [
    "however", "limitation", "limitations", "lack of", "future work",
    "unexplored", "further research", "requires further", "not addressed",
    "not investigated", "not evaluated", "remains unclear", "remains challenging",
    "open problem", "needs further study"
]

def detect_gap_indicators(sentence: str) -> list[str]:
    sent_lower = sentence.lower()
    return [cue for cue in GAP_INDICATORS if cue in sent_lower]

def rule_score(sentence: str, precomputed_matches: list[str] = None) -> float:
    matches = precomputed_matches if precomputed_matches is not None else detect_gap_indicators(sentence)
    return min(len(matches) / 2.0, 1.0) if matches else 0.0