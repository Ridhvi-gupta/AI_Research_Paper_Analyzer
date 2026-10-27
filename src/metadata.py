def get_metadata(text: str, filename: str, page_count: int, sentences: list[str] = None) -> dict:
    words = text.split()
    sent_count = len(sentences) if sentences is not None else len(text.split("."))
    return {
        "paper": filename,
        "pages": page_count,
        "words": len(words),
        "sentences": sent_count
    }