"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

import pytest
from inflammation.models import daily_mean, daily_max, daily_min, patient_normalise

@pytest.mark.parametrize(
    "test_input, test_result", 
    [
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([4, 5])),
        (np.array([[-1, -2], [-3, -4], [-5, -6]]), np.array([-3, -4]))
    ])
def test_daily_mean_parametrized(test_input, test_result):
    """Test daily mean function works for input arrays of zeros, positive and negative integers."""
    npt.assert_array_equal(daily_mean(test_input), test_result)

@pytest.mark.parametrize(
    "test_input, test_result",
    [
        (np.array([[1, 2, 12], [3, 4, 34], [5, 6, 1]]), np.array([5, 6, 34])),
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        (np.array([[-1, -2], [-3, -4], [-5, -6]]), np.array([-1, -2]))
    ])  
def test_daily_max_parametrized(test_input, test_result):
    """Test daily max function works for input arrays of zeroes, positive and negative integers."""
    npt.assert_array_equal(daily_max(test_input), test_result)  

@pytest.mark.parametrize(
    "test_input, test_result",
    [
        (np.array([[1, 2], [-3, 4], [5, -6]]), np.array([-3, -6])),
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0]))
    ])
def test_daily_min_parametrized(test_input, test_result):
    """Test daily min function works for input arrays of positive and negative integers."""
    npt.assert_array_equal(daily_min(test_input), test_result)  

def test_daily_min_string():
    """Test that min function raises an error for an array of strings."""

    test_input = np.array([['a', 'b'],
                           ['c', 'd'],
                           ['e', 'f']])

    with pytest.raises(TypeError):
        daily_min(test_input)

@pytest.mark.parametrize(
    "test_input, test_result, expect_raises, error_message",
    [
        ([
            [1, 2, 3],
            [4, float('nan'), 6],
            [7, 8, 9]
            ], 
            [
            [0.33, 0.67, 1],
            [0.67, 0, 1],
            [0.78, 0.89, 1]
            ],
            None,
            None
        ),
        (
            [0, 0, 0],
            None,
            ValueError,
            'Input data must be a 2D array'   
        ),
        ([
            [-1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
            ],
            [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
            ],
            ValueError,
            'Negative values not allowed in input data'
        ),
        (
            "dog",
            None,
            TypeError,
            'Input data must be a Numpy array'
        )
    ])
def test_patient_normalise_parametrized(test_input, test_result, expect_raises, error_message):
    """Test patient normalise function works for input arrays of positive integers.
        Test with a relative and absolute tolerance of 0.01.
    """
    if isinstance(test_input, list):
        test_input = np.array(test_input)

    if expect_raises is not None:
        with pytest.raises(expect_raises, match=error_message):
            patient_normalise(test_input)
    else:
        result = patient_normalise(test_input)
        npt.assert_allclose(result, np.array(test_result), rtol=1e-2, atol=1e-2)