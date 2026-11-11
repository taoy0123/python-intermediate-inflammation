"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest

import pytest
from inflammation.models import daily_mean, daily_max, daily_min, patient_normalise, compute_standard_deviation, analyse_data, CSVDataSource, JSONDataSource
import os
from unittest.mock import Mock

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

def test_analyse_data_mock_source():
  data_source = Mock()
  data_source.load_inflammation_data.return_value = [
    [[1, 2, 3], [4, 5, 6]],  # Mock data for the first dataset
    [[7, 8, 9], [10, 11, 12]]  # Mock data for the second dataset
  ]
  analyse_data(data_source)

def test_analyse_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data")
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)
    expected = [0.,0.22510286,0.18157299,0.1264423,0.9495481,0.27118211,
                0.25104719,0.22330897,0.89680503,0.21573875,1.24235548,0.63042094,
                1.57511696,2.18850242,0.3729574,0.69395538,2.52365162,0.3179312,
                1.22850657,1.63149639,2.45861227,1.55556052,2.8214853,0.92117578,
                0.76176979,2.18346188,0.55368435,1.78441632,0.26549221,1.43938417,
                0.78959769,0.64913879,1.16078544,0.42417995,0.36019114,0.80801707,
                0.50323031,0.47574665,0.45197398,0.22070227]
    npt.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "test_input, test_result",
    [
        ([[[1, 2, 3], [4, 5, 6]],[[7,8,9],[10,11,12]]], [3., 3., 3.]),
        ([[[1,2,3],[4,5,6]]], [0, 0, 0]),
        ([[[1,2,3]], [[4,5,6]]], [1.5, 1.5, 1.5])
    ],
    ids=["two files, each with two patients", "two patients in one file", "two patients in different files"]
)
def test_compute_standard_deviation_parametrized(test_input, test_result):
    """Test compute_standard_deviation function works for input arrays of positive integers."""
    result = compute_standard_deviation(test_input)
    npt.assert_array_almost_equal(result, test_result)

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
