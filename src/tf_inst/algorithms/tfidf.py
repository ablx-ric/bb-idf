from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TfidfVectorizerWrapper:
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)
        self._vocabulary = None

    def fit(self, documents: list[str]):
        self.vectorizer.fit(documents)
        self._vocabulary = self.vectorizer.vocabulary_
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        return self.vectorizer.transform(documents).toarray()

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        result = self.vectorizer.fit_transform(documents)
        self._vocabulary = self.vectorizer.vocabulary_
        return result.toarray()

    @property
    def vocabulary(self) -> dict:
        return self._vocabulary

    @property
    def feature_names(self) -> list[str]:
        return self.vectorizer.get_feature_names_out().tolist()
