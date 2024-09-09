import pytest


@pytest.mark.parametrize("a,b,result", [(1, 1, 2), (2, 3, 5)])
def test_add(a, b, result):
    assert a + b == result


def test_simple_add(a, b, result):
    assert a + b == result
