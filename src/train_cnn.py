import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

DATASET_PATH = "data/dataset/gap_dataset.csv"
MODEL_DIR = "models"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_gap_model.keras")
MAX_WORDS = 10000
MAX_LENGTH = 50

def train():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing dataset at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    texts = df["sentence"].astype(str).values
    labels = df["label"].values

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    padded = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LENGTH, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128),
        Conv1D(filters=128, kernel_size=5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Model Trained. Test Accuracy: {acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Saved model to {MODEL_PATH} and tokenizer to {TOKENIZER_PATH}")

if __name__ == "__main__":
    train()