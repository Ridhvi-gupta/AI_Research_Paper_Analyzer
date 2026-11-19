import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join("data", "dataset", "gap_dataset.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_gap_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

VOCAB_SIZE = 5000
EMBEDDING_DIM = 128
MAX_LEN = 60

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Safely load CSV with quotes handling
    df = pd.read_csv(DATASET_PATH, quotechar='"', skipinitialspace=True)
    
    # Auto-detect sentence and label columns
    sent_col = [c for c in df.columns if "sent" in c.lower() or "text" in c.lower()][0]
    label_col = [c for c in df.columns if "label" in c.lower() or "target" in c.lower()][0]

    sentences = df[sent_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()

    print(f"Loaded {len(sentences)} balanced academic samples for training.")

    # 1. Tokenization & Padding
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    X = np.array(padded)
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. 1D-CNN Topology
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nTraining 1D-CNN Sequence Classifier...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=8,
        verbose=1
    )

    # 3. Validation Evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print("\n" + "="*50)
    print(f"1D-CNN Training Complete!")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Loss:     {val_loss:.4f}")
    print("="*50)

    # 4. Save Model & Tokenizer
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"\n✓ Saved model weights to: {MODEL_PATH}")
    print(f"✓ Saved tokenizer to:     {TOKENIZER_PATH}")

if __name__ == "__main__":
    train()