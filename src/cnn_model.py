import os
import pickle
import re
import numpy as np

_LOADED_MODEL = None
_LOADED_TOKENIZER = None

MODEL_PATH = os.path.join("models", "cnn_gap_model.keras")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")

def load_cnn_gap_model():
    """Loads the pre-trained Keras model and tokenizer from disk if available."""
    global _LOADED_MODEL, _LOADED_TOKENIZER
    if _LOADED_MODEL is not None and _LOADED_TOKENIZER is not None:
        return _LOADED_MODEL, _LOADED_TOKENIZER

    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        try:
            import tensorflow as tf
            _LOADED_MODEL = tf.keras.models.load_model(MODEL_PATH)
            with open(TOKENIZER_PATH, "rb") as f:
                _LOADED_TOKENIZER = pickle.load(f)
            return _LOADED_MODEL, _LOADED_TOKENIZER
        except Exception:
            return None, None
    return None, None

def predict_gap_probability(sentence: str) -> float:
    """
    Evaluates academic sentence sequence structure:
    Detects contrastive transitions, failure modes, scale degradation, and future projections.
    """
    model, tokenizer = load_cnn_gap_model()

    if model is not None and tokenizer is not None:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = tokenizer.texts_to_sequences([sentence])
            padded = pad_sequences(seq, maxlen=60, padding='post', truncating='post')
            pred = float(model.predict(padded, verbose=0)[0][0])
            return round(pred, 3)
        except Exception:
            pass

    # Structural Syntactic Concession Evaluator (Domain-Agnostic)
    s_low = sentence.lower()
    score = 0.15

    # 1. Structural contrast patterns (e.g. "although X, Y fails", "despite X, Y cannot")
    if re.search(r'\b(although|despite|in spite of|nevertheless|yet|whereas)\b', s_low):
        score += 0.25
    if re.search(r'\b(however|nonetheless|conversely)\b', s_low):
        score += 0.20

    # 2. Deficit & Failure patterns (e.g. "fails to", "inability to", "degrades rapidly", "does not scale")
    if re.search(r'\b(fails? to|cannot|unable to|inability|degrades?|deteriorates?|suboptimal)\b', s_low):
        score += 0.30

    # 3. Resource & Complexity bottlenecks (e.g. "computationally prohibitive", "memory bound", "lack of")
    if re.search(r'\b(prohibitive|unviable|bottleneck|quadratic|expensive|scarcity|dearth|lack of)\b', s_low):
        score += 0.30

    # 4. Explicit limitation/future assertions
    if re.search(r'\b(limitation|drawback|future work|unaddressed|unresolved|open question)\b', s_low):
        score += 0.35

    return min(0.98, round(score, 3))