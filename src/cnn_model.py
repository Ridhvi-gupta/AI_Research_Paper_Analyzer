import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "models/cnn_gap_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
MAX_LENGTH = 50

def load_cnn():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)):
        raise FileNotFoundError("Model or tokenizer not found. Run train_cnn.py first.")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_gap_probability(model, tokenizer, sentence: str) -> float:
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    return float(model.predict(padded, verbose=0)[0][0])