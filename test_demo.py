import pytest
from demo import run

@pytest.fixture(scope="module")
def results():
    return dict(run())

@pytest.mark.parametrize("scenario, expected", [
    ("baseline", [1, 1, 1]),
    ("hallucination", [1, 0, 1]),
    ("irrelevant retrieval", [0, 1, 0]),
])
def test_faults_are_detected(results, scenario, expected):
    assert results[scenario] == expected
