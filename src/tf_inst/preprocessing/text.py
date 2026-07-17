import re
import string


class Preprocessor:
    def __init__(self, locale: str = "es"):
        self.locale = locale

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        # TODO: cargar stopwords según locale
        stopwords = set()
        return [t for t in tokens if t not in stopwords]

    def stem(self, tokens: list[str]) -> list[str]:
        # TODO: aplicar stemming (p.ej. snowballstemmer)
        return tokens

    def __call__(self, text: str) -> str:
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        return " ".join(tokens)
