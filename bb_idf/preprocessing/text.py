import functools

import spacy


@functools.cache
def _load_nlp():
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    nlp.max_length = 20_000_000
    return nlp


class Preprocessor:
    def __init__(self, locale: str = "es"):
        self.locale = locale
        self._nlp = _load_nlp()

    def tokenize(self, text: str) -> list[str]:
        doc = self._nlp(text)
        return [t.lower_ for t in doc if not t.is_punct and not t.is_space]

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return tokens
        doc = self._nlp(" ".join(tokens))
        return [t.text for t in doc if not t.is_stop]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return tokens
        doc = self._nlp(" ".join(tokens))
        return [t.lemma_.lower() for t in doc]

    def __call__(self, text: str) -> str:
        doc = self._nlp(text)
        tokens = [
            t.lemma_.lower() for t in doc
            if not t.is_punct and not t.is_space and not t.is_stop
            and not t.like_num and len(t.text) >= 3
        ]
        return " ".join(tokens)
