import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

STOPWORDS = {
    "the", "and", "is", "in", "to", "of", "for", "with", "that", "this", "on", "as",
    "are", "by", "an", "be", "we", "from", "which", "our", "were", "been", "have",
    "has", "had", "they", "their", "more", "such", "than", "also", "into", "can",
    "could", "would", "should", "using", "used", "paper", "method", "proposed"
}

def get_clean_tokens(text: str) -> list[str]:
    """Tokenizes and cleans text, removing academic stopwords and short tokens."""
    tokens = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def get_word_frequency(text: str, top_n: int = 20) -> list[tuple[str, int]]:
    """Returns top N most frequent non-stopword tokens."""
    tokens = get_clean_tokens(text)
    return Counter(tokens).most_common(top_n)

def get_word_frequency_dataframe(text: str, top_n: int = 20) -> pd.DataFrame:
    """Returns top word frequencies as a formatted pandas DataFrame."""
    freq = get_word_frequency(text, top_n)
    return pd.DataFrame(freq, columns=["Keyword", "Frequency"])

def create_word_frequency_chart(text: str, top_n: int = 15, bg_color: str = "#080c14", text_color: str = "#f8fafc"):
    """Generates a styled horizontal bar chart of top corpus keywords."""
    data = get_word_frequency(text, top_n)
    if not data:
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        ax.text(0.5, 0.5, "No textual data available", ha="center", color=text_color)
        return fig

    words, counts = zip(*data)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    ax.barh(words[::-1], counts[::-1], color="#ff7700", edgecolor="#ffe066", linewidth=1.2, alpha=0.92)
    ax.tick_params(colors=text_color, labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#ff7700")
    ax.spines['bottom'].set_color("#ff7700")
    ax.set_title(f"Top {top_n} Frequent Research Terms", color=text_color, fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig

def create_wordcloud(text: str, bg_color: str = "white"):
    """Generates a high-resolution WordCloud image from the document text."""
    clean_text = " ".join(get_clean_tokens(text))
    wc = WordCloud(
        width=1000,
        height=450,
        background_color=bg_color,
        colormap="YlOrBr" if bg_color == "black" else "viridis",
        max_words=100
    ).generate(clean_text or "research paper analysis")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig

def get_sentence_length_distribution(sentences: list[str]) -> list[int]:
    """Returns a list of word counts per sentence for syntactic length profiling."""
    return [len(s.split()) for s in sentences if len(s.split()) > 0]
