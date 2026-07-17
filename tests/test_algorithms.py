import pytest
import numpy as np
from tf_inst.algorithms import TfidfVectorizerWrapper, TextRankVectorizer, TFPDC_Scalable


class TestTfidfVectorizerWrapper:
    def test_fit_transform_shape(self):
        docs = ["hola mundo", "mundo adiós"]
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
        docs = ["hola mundo cruel", "mundo adiós"]
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
        X = vec.transform(["término desconocido"])
        assert np.all(X == 0)

    def test_transform_without_fit_raises(self):
        vec = TextRankVectorizer()
        with pytest.raises(ValueError):
            vec.transform(["hola"])


class TestTFPDCScalable:
    def test_fit_transform_returns_dense_array(self):
        docs = ["hola mundo cruel", "mundo adiós"]
        vec = TFPDC_Scalable()
        X = vec.fit_transform(docs)
        assert isinstance(X, np.ndarray)
        assert X.shape == (2, 4)

    def test_vocabulary(self):
        docs = ["hola mundo"]
        vec = TFPDC_Scalable()
        vec.fit(docs)
        assert "hola" in vec.vocabulary

    def test_aof_filtering_removes_frequent_terms(self):
        docs = ["común raro", "común", "común", "común"]
        vec = TFPDC_Scalable(max_threshold=0.5)
        vec.fit(docs)
        assert "común" not in vec.vocabulary
        assert "raro" in vec.vocabulary

    def test_transform_uses_fitted_vocabulary(self):
        vec = TFPDC_Scalable()
        vec.fit(["hola mundo", "mundo adiós"])
        X = vec.transform(["hola desconocido"])
        assert X.shape == (1, len(vec.vocabulary))

    def test_transform_without_fit_raises(self):
        vec = TFPDC_Scalable()
        with pytest.raises(ValueError):
            vec.transform(["hola"])
