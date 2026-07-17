from tf_inst.preprocessing import Preprocessor


class TestPreprocessor:
    def test_tokenize_lowercase(self):
        p = Preprocessor()
        tokens = p.tokenize("Hola Mundo!")
        assert tokens == ["hola", "mundo"]

    def test_call_returns_string(self):
        p = Preprocessor()
        result = p("Hola Mundo!")
        assert isinstance(result, str)
