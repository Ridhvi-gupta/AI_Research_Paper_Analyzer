from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def get_word_frequency(text: str, top_n: int = 20) -> list[tuple[str, int]]:
    return Counter(text.split()).most_common(top_n)

def create_word_frequency_chart(text: str, top_n: int = 20):
    data = get_word_frequency(text, top_n)
    if not data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No text available", ha="center")
        return fig
    words, counts = zip(*data)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(words, counts, color="#2b5c8f")
    ax.set_title(f"Top {top_n} Frequent Keywords", fontsize=12)
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def create_wordcloud(text: str):
    wc = WordCloud(width=1000, height=450, background_color="white", colormap="viridis").generate(text or "empty")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig