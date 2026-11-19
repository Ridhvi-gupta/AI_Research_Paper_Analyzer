import re
from src.gap_rules import detect_gap_indicators, rule_score, load_learned_triggers
from src.cnn_model import predict_gap_probability, batch_predict_gap_probabilities

def analyze_sentence(sentence: str, confidence_threshold: float = 0.38) -> dict:
    """
    Analyzes an individual sentence using hybrid NLP rules and 1D-CNN sequence modeling.
    Returns matched cues, probability metrics, and candidate status.
    """
    indicators = detect_gap_indicators(sentence)
    nlp_sc = rule_score(sentence, indicators)
    cnn_sc = predict_gap_probability(sentence)

    # Hybrid confidence scoring
    if indicators and cnn_sc >= 0.30:
        combined_sc = round(0.5 * nlp_sc + 0.5 * cnn_sc, 3)
    elif indicators:
        combined_sc = round(0.6 * nlp_sc + 0.4 * cnn_sc, 3)
    else:
        combined_sc = round(cnn_sc, 3)

    is_candidate = combined_sc >= confidence_threshold

    return {
        "sentence": sentence,
        "indicators": indicators,
        "nlp_score": round(nlp_sc, 2),
        "cnn_score": round(cnn_sc, 2),
        "combined_score": combined_sc,
        "potential_gap": is_candidate
    }

def batch_detect_gaps(sentences: list[str], doc_name: str = "Paper", threshold: float = 0.38) -> list[dict]:
    """
    Fast vectorized evaluation for all sentences in a document.
    """
    if not sentences:
        return []

    cnn_scores = batch_predict_gap_probabilities(sentences)
    active_triggers = load_learned_triggers()
    gaps = []

    for idx, (sent, cnn_sc) in enumerate(zip(sentences, cnn_scores)):
        s_low = sent.lower()
        matched = [w for w in active_triggers if w in s_low]
        nlp_sc = min(1.0, len(matched) * 0.40) if matched else 0.0

        if matched and cnn_sc >= 0.30:
            hybrid_sc = round(0.5 * nlp_sc + 0.5 * cnn_sc, 3)
        elif matched:
            hybrid_sc = round(0.6 * nlp_sc + 0.4 * cnn_sc, 3)
        else:
            hybrid_sc = round(cnn_sc, 3)

        if hybrid_sc >= threshold:
            gaps.append({
                "id": f"{doc_name}_{hash(sent)}",
                "doc_name": doc_name,
                "sentence": sent,
                "indicators": matched,
                "rule_score": nlp_sc,
                "cnn_score": cnn_sc,
                "hybrid_score": hybrid_sc
            })

    gaps.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return gaps
