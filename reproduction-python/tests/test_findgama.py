import importlib.util
import io
import random
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy


def load_pretreatment():
    # This test never trains MLkNN or loads NLP models/corpora.
    stubs = {
        "gensim": {"corpora": object()},
        "gensim.models": {"LdaModel": object()},
        "mlknn": {"MLkNN": lambda **kwargs: None},
        "nltk.corpus": {"stopwords": object()},
        "nltk.tokenize": {"word_tokenize": object()},
        "nltk.stem.porter": {"PorterStemmer": object()},
    }
    modules = {}
    for name, attrs in stubs.items():
        modules[name] = ModuleType(name)
        vars(modules[name]).update(attrs)
    path = Path(__file__).resolve().parents[1] / "Pretreatment.py"
    spec = importlib.util.spec_from_file_location("devrec_pretreatment_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


BR_MLkNN = load_pretreatment().BR_MLkNN


class FindGamaTest(unittest.TestCase):
    def search(self, score):
        model = BR_MLkNN()
        evaluated = {5: [], 10: []}

        def test(X, y, recall, gama):
            value = score(recall, gama)
            evaluated[recall].append((tuple(gama), value))
            return value

        args = object(), object(), object(), object()
        with patch.object(model, "set") as fit, patch.object(model, "test", test):
            with redirect_stdout(io.StringIO()):
                model.findgama(*args)
            fit.assert_called_once_with(*args[2:])
        return model, evaluated

    def test_positive_start_can_keep_improving_to_last_candidate(self):
        model, evaluated = self.search(lambda recall, vector: vector[0])
        for recall, best in ((5, model.gama5), (10, model.gama10)):
            initial = evaluated[recall][0][0]
            self.assertGreater(initial[0], 0)
            self.assertLess(initial[0] + 2 * 0.03, 1.0)
            self.assertEqual(best[0], 1.0)
            self.assertIn((tuple(best), 1.0), evaluated[recall])
            self.assertTrue(all(0 <= value <= 1 for row, _ in evaluated[recall]
                                for value in row))

    def test_zero_score_plateau_stops_and_keeps_evaluated_start(self):
        model, evaluated = self.search(lambda recall, vector: 0.0)
        for recall, best in ((5, model.gama5), (10, model.gama10)):
            # Each of ten starts is scored once, then rejected once per weight.
            self.assertEqual(len(evaluated[recall]), 10 * (1 + 5))
            self.assertEqual(tuple(best), evaluated[recall][0][0])
            self.assertNotEqual(evaluated[recall][0][0], evaluated[recall][1][0])

    def test_best_vectors_are_independent_and_reproducible(self):
        initial_rng_state = random.getstate()

        def score(recall, vector):
            return vector[0] if recall == 5 else vector[1]

        model, evaluated = self.search(score)
        with patch.object(random, "random", side_effect=AssertionError("global RNG used")):
            repeated, repeated_evaluated = self.search(score)
        self.assertEqual(random.getstate(), initial_rng_state)
        self.assertEqual(evaluated, repeated_evaluated)
        self.assertEqual(model.gama5, repeated.gama5)
        self.assertEqual(model.gama10, repeated.gama10)
        self.assertIsNot(model.gama5, model.gama10)
        for recall, best in ((5, model.gama5), (10, model.gama10)):
            self.assertIn((tuple(best), max(value for _, value in evaluated[recall])),
                          evaluated[recall])
        self.assertEqual(model.gama5[0], 1.0)
        self.assertLess(model.gama5[1], 1.0)
        self.assertEqual(model.gama10[1], 1.0)
        self.assertLess(model.gama10[0], 1.0)


class ScoringTest(unittest.TestCase):
    def test_repeated_scoring_keeps_validation_data_and_fitted_terms(self):
        model = BR_MLkNN()
        model.br_mlknn = SimpleNamespace(predict_proba=lambda X: numpy.array([[0.3, 0.7]]))
        model.termlenth = model.topiclenth = model.productlenth = model.componentlenth = 1
        model.dl_term = numpy.array([[2.0], [4.0]])
        model.dl_topic = numpy.array([[0.75], [0.25]])
        model.dl_product = numpy.array([[0.1], [0.2]])
        model.dl_component = numpy.array([[0.3], [0.1]])
        X = numpy.array([[3.0, 0.4, 1.0, 1.0]])
        before_X = X.copy()
        before_term = model.dl_term.copy()
        gama = [0.2, 0.3, 0.4, 0.5, 0.6]
        model.recall = lambda y, score, n: score.copy()
        with redirect_stdout(io.StringIO()):
            first = model.test(X, None, 5, gama=gama)
            second = model.test(X, None, 10, gama=gama)
        numpy.testing.assert_allclose(first, [[0.41, 0.465]])
        numpy.testing.assert_allclose(second, first)
        numpy.testing.assert_array_equal(X, before_X)
        numpy.testing.assert_array_equal(model.dl_term, before_term)


if __name__ == "__main__":
    unittest.main()
