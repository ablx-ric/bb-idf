import numpy as np


class CustomVectorizer:
    """Placeholder para tu algoritmo propuesto.

    Debe implementar la misma interfaz que TfidfVectorizerWrapper
    para poder ejecutar el benchmark de forma intercambiable.
    """

    def __init__(self, **kwargs):
        self._vocabulary = None

    def fit(self, documents: list[str]):
        # TODO: implementar lógica de entrenamiento
        self._vocabulary = {}
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        # TODO: transformar documentos en matriz documentos x términos
        return np.zeros((len(documents), 1))

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    @property
    def vocabulary(self) -> dict:
        return self._vocabulary
