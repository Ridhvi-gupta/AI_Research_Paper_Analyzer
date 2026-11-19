import os
import re
from datetime import datetime

def compute_readability_metrics(text: str) -> dict:
    """
    Computes academic readability statistics:
    - Average sentence length
    - Type-Token Ratio (lexical diversity)
    - Approximate Flesch Reading Ease score
    """
    words = [w.lower() for w in re.findall(r'\b[A-Za-z]{2,}\b', text)]
    if not words:
        return {"avg_sentence_len": 0, "lexical_diversity": 0.0, "flesch_score": 0.0}

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip().split()) >= 3]
    num_sents = max(1, len(sentences))
    num_words = len(words)
    unique_words = len(set(words))

    avg_sent_len = round(num_words / num_sents, 1)
    lexical_diversity = round(unique_words / num_words, 3)

    # Syllable approximation for Flesch score
    def count_syllables(word):
        w = word.lower()
        count = len(re.findall(r'[aeiouy]+', w))
        if w.endswith('e') and not w.endswith('le') and count > 1:
            count -= 1
        return max(1, count)

    total_syllables = sum(count_syllables(w) for w in words[:1500])
    sample_words = min(num_words, 1500)
    asl = avg_sent_len
    asw = total_syllables / max(1, sample_words)
    flesch = round(206.835 - (1.015 * asl) - (84.6 * asw), 1)

    return {
        "avg_sentence_len": avg_sent_len,
        "lexical_diversity": lexical_diversity,
        "flesch_score": max(0.0, min(100.0, flesch))
    }

def get_metadata(text: str, filename: str, page_count: int = 1, sentences: list[str] = None) -> dict:
    """
    Extracts rich bibliometric and textual metadata from an ingested paper.
    """
    ext = os.path.splitext(filename)[1].upper().replace(".", "")
    words = text.split()
    sent_list = sentences if sentences is not None else [s.strip() for s in text.split(".") if s.strip()]

    # Extract potential paper title from first prominent line
    first_lines = [l.strip() for l in text.split("\n")[:8] if len(l.strip().split()) >= 3]
    inferred_title = first_lines[0] if first_lines else os.path.splitext(filename)[0]

    readability = compute_readability_metrics(text)

    return {
        "paper": filename,
        "inferred_title": inferred_title[:120],
        "format": ext or "PDF",
        "pages": max(1, page_count),
        "words": len(words),
        "sentences": len(sent_list),
        "characters": len(text),
        "lexical_diversity": readability["lexical_diversity"],
        "avg_sentence_length": readability["avg_sentence_len"],
        "readability_flesch": readability["flesch_score"],
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
