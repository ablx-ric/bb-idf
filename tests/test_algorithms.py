import pytest
import numpy as np
from bb_idf.algorithms import TfidfVectorizerWrapper, TextRankVectorizer, BBIDF


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

    def test_no_hard_zeroing(self):
        """Definicion validada: los terminos fuera de banda NO se anulan.

        Solo cambia el IDF (df -> df_banda); todo termino presente (tf > 0)
        conserva peso positivo.
        """
        docs = ["turismo turismo naturaleza",
                 "naturaleza naturaleza turismo"]
        vec = BBIDF()
        X = vec.fit_transform(docs)
        assert X[0, vec._vocabulary["turismo"]] > 0
        assert X[0, vec._vocabulary["naturaleza"]] > 0
        assert X[1, vec._vocabulary["naturaleza"]] > 0
        assert X[1, vec._vocabulary["turismo"]] > 0

    def test_matches_experiment_definition(self):
        """La clase empaquetada y scorers.bbidf_weights producen lo mismo."""
        from bb_idf.experiment import scorers

        docs = ["turismo turismo naturaleza aventura turismo",
                 "ruinas ruinas arqueologia ruinas historia",
                 "turismo ruinas historia"]
        vec = BBIDF()
        W_class = vec.fit_transform(docs)
        X_counts = vec.vectorizer.transform(docs).toarray()
        W_ref = scorers.bbidf_weights(X_counts)
        assert np.allclose(W_class, W_ref)

    def test_idf_uses_smooth_formula(self):
        """idf(t) = ln((1+N)/(1+df_banda(t))) + 1."""
        docs = ["hola mundo", "hola hola"]
        vec = BBIDF()
        vec.fit(docs)
        n_docs = len(docs)
        for term, idx in vec._vocabulary.items():
            expected = np.log((1.0 + n_docs) / (1.0 + vec._df_banda[idx])) + 1.0
            assert np.isclose(vec._idf[idx], expected)
