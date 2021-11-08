from src.utils.evaluator import Evaluator


def test_evaluator(classification_model):

    evaluator = Evaluator()
    accuracy = evaluator.evaluate(classification_model)

    assert 0 < accuracy < 1, f"incorrect accuracy value"
