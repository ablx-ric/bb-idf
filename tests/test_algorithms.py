import pytest
import numpy as np
from tf_inst.algorithms import TfidfVectorizerWrapper, TextRankVectorizer, BBIDF


class TestTfidfVectorizerWrapper:
    def test_fit_transform_shape(self):
        docs = ["hola mundo", "mundo adios"]
        vec = TfidfVectorizerWrapper()
        X = vec.fit_transform(docs)
        assert X.shape == (2, 3)

    def test_vocabulary(self):
        docs = ["hola mundo"]
        vec = TfidfVectorizerWrapper()
        vec.fit(docs)
        assert "hola" in vec.vocabulary


class TestTextRankVectorizer:
    def test_fit_transform_shape(self):
        docs = ["hola mundo cruel", "mundo adios"]
        vec = TextRankVectorizer()
        X = vec.fit_transform(docs)
        assert X.shape == (2, 4)

    def test_vocabulary(self):
        docs = ["hola mundo"]
        vec = TextRankVectorizer()
        vec.fit(docs)
        assert "hola" in vec.vocabulary

    def test_scores_positive_for_present_terms(self):
        docs = ["el gato come pescado y el gato duerme"]
        vec = TextRankVectorizer()
        X = vec.fit_transform(docs)
        j = vec.vocabulary["gato"]
        assert X[0, j] > 0

    def test_transform_ignores_oov_terms(self):
        vec = TextRankVectorizer()
        vec.fit(["hola mundo"])
        X = vec.transform(["termino desconocido"])
        assert np.all(X == 0)

    def test_transform_without_fit_raises(self):
        vec = TextRankVectorizer()
        with pytest.raises(ValueError):
            vec.transform(["hola"])


class TestBBIDF:
    def test_fit_transform_returns_dense_array(self):
        docs = ["hola mundo cruel", "mundo adios"]
        vec = BBIDF()
        X = vec.fit_transform(docs)
        assert isinstance(X, np.ndarray)
        assert X.shape == (2, 4)

    def test_vocabulary(self):
        docs = ["hola mundo"]
        vec = BBIDF()
        vec.fit(docs)
        assert "hola" in vec.vocabulary

    def test_idf_values_are_finite_and_positive(self):
        vec = BBIDF()
        docs = ["hola mundo raro", "hola mundo", "hola mundo"]
        vec.fit(docs)
        for term in ["hola", "mundo", "raro"]:
            idx = vec._vocabulary[term]
            assert np.isfinite(vec._idf[idx])
            assert vec._idf[idx] > 0

    def test_transform_uses_fitted_vocabulary(self):
        vec = BBIDF()
        vec.fit(["hola mundo", "mundo adios"])
        X = vec.transform(["hola desconocido"])
        assert X.shape == (1, len(vec.vocabulary))

    def test_transform_without_fit_raises(self):
        vec = BBIDF()
        with pytest.raises(ValueError):
            vec.transform(["hola"])

    def test_self_similarity_is_max(self):
        docs = ["turismo naturaleza aventura", "ruinas arqueologia historia"]
        vec = BBIDF()
        X = vec.fit_transform(docs)
        assert X[0, vec._vocabulary["turismo"]] > 0
        assert X[1, vec._vocabulary["ruinas"]] > 0
