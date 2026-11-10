"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize('data, expected_standard_deviation, expected_raises, error_message',
[
    ([[0, 0, 0]], 0.0, None, None),
    ([[1.0, 1.0, 1.0]], 0, None, None),
    ([1, 2, 3], 1.0, ValueError, "Input data must be a 2D array."),
    ([], 0, ValueError, "Input data must be a 2D array."),
    ([[1,2,3], [4,5,6]], 0.816496580927726, None, None),
    ([[-1, -2, -3], [-4, -5, -6]], 0.816496580927726, None, None)
],
ids=['all zeros', 'all ones', '1D array', 'empty list', '2D array', 'negative values'])
def test_daily_standard_deviation(data, expected_standard_deviation, expected_raises, error_message):
    from inflammation.models import daily_std
    if expected_raises is None:
        result_data = daily_std(data)
        npt.assert_approx_equal(result_data, expected_standard_deviation)
    else:
        with pytest.raises(expected_raises, match=error_message):
            daily_std(data)
