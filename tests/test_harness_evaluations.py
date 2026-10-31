import pytest

from automl_architect.evaluations import evaluate


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_live_evaluation_requires_explicit_opt_in(provider):
    with pytest.raises(ValueError, match="--live"):
        evaluate(provider=provider)
