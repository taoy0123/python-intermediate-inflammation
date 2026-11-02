"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

import pytest
from inflammation.models import daily_mean, daily_max, daily_min

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_max_integers():
    """Test that max function works for an array of positive integers."""

    test_input = np.array([[1, 2, 12],
                           [3, 4, 34],
                           [5, 6, 1]])
    test_result = np.array([5, 6, 34])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)

def test_daily_min_integers():
    """Test that min function works for an array of positive and negative integers."""

    test_input = np.array([[1, 2],
                           [-3, 4],
                           [5, -6]])
    test_result = np.array([-3, -6])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)

def test_daily_min_string():
    """Test that min function raises an error for an array of strings."""

    test_input = np.array([['a', 'b'],
                           ['c', 'd'],
                           ['e', 'f']])

    with pytest.raises(TypeError):
        daily_min(test_input)