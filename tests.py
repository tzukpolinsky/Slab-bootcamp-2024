import pytest


@pytest.mark.parametrize("a,b,result", [(1, 1, 2), (2, 3, 5)])
def test_add(a, b, result):
    assert a + b == result


def test_simple_add(a, b, result):
    assert a + b == result



def process_data(df, threshold):
    """
    Filters the input DataFrame by removing rows where values are below a specified threshold.

    :param df: pandas DataFrame containing numerical data.
    :param threshold: A numeric value used to filter the rows. Rows where all values are less than the threshold
                      will be removed.
    :return: A new DataFrame containing only the rows where values exceed the threshold in at least one column.

    Edge cases:
    - If no rows exceed the threshold, an empty DataFrame is returned.
    - If `threshold` is None or invalid, raises a ValueError.

    Example:
    >>> process_data(df, 10)
    Filters rows where values are greater than 10.
    """