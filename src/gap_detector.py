from src.gap_rules import detect_gap_indicators, rule_score
from src.cnn_model import predict_gap_probability

def analyze_sentence(sentence: str, model=None, tokenizer=None) -> dict:
    indicators = detect_gap_indicators(sentence)
    nlp_sc = rule_score(sentence, indicators)

    if model is not None and tokenizer is not None:
        cnn_sc = predict_gap_probability(model, tokenizer, sentence)
        combined_sc = 0.4 * nlp_sc + 0.6 * cnn_sc
        is_candidate = combined_sc >= 0.5
    else:
        cnn_sc = 0.0
        combined_sc = nlp_sc
        is_candidate = bool(indicators)

    return {
        "sentence": sentence,
        "indicators": indicators,
        "nlp_score": round(nlp_sc, 2),
        "cnn_score": round(cnn_sc, 2) if model else "N/A",
        "combined_score": round(combined_sc, 2),
        "potential_gap": is_candidate
    }