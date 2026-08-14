import numpy as np
import pytest

from bb_idf.experiment import similarity
from bb_idf.experiment import stats


class TestRBO:
    def test_identical_lists_give_one(self):
        assert similarity.rbo_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_disjoint_lists_give_zero(self):
        assert similarity.rbo_at_k(["a", "b"], ["c", "d"], k=4) == 0.0

    def test_rbo_penalizes_order_swap(self):
        a = ["x", "y", "z"]
        b = ["y", "x", "z"]
        # Same set, different order at the top: RBO must be < 1.
        assert similarity.rbo_at_k(a, b, k=3) < 1.0

    def test_rbo_is_symmetric(self):
        a = ["x", "y", "z"]
        b = ["y", "z", "x"]
        assert similarity.rbo_at_k(a, b, k=3) == similarity.rbo_at_k(b, a, k=3)

    def test_rbo_bounded_in_unit_interval(self):
        for k in (5, 10, 20, 50):
            v = similarity.rbo_at_k(list("abcdefghijklmnopqrstuvwxyz"),
                                    list("zyxwvutsrqponmlkjihgfedcba"), k=k)
            assert 0.0 <= v <= 1.0


class TestJaccard:
    def test_jaccard_ignores_order(self):
        assert similarity.jaccard_at_k(["a", "b", "c"], ["c", "b", "a"], k=3) == 1.0

    def test_jaccard_disjoint(self):
        assert similarity.jaccard_at_k(["a", "b"], ["c", "d"], k=4) == 0.0


class TestHolm:
    def test_all_nonsignificant(self):
        out = stats.holm_bonferroni([1.0, 1.0, 1.0])
        assert np.allclose(out, [1.0, 1.0, 1.0])

    def test_known_values(self):
        out = stats.holm_bonferroni([0.01, 0.02, 0.5])
        # n=3: 0.01*3=0.03, 0.02*2=0.04, 0.5*1=0.5 -> cumulative max in order.
        assert np.allclose(out, [0.03, 0.04, 0.5])

    def test_preserves_input_order(self):
        p = [0.2, 0.01, 0.5, 0.03]
        out = stats.holm_bonferroni(p)
        assert len(out) == len(p)
        # The smallest raw p stays smallest adjusted.
        assert out[int(np.argmin(p))] == np.min(out)
