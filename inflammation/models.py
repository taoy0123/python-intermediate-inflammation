"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    :return: 2D Numpy array of data
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array.

    :param data: 2D array of inflammation data
    :return: 1D array of daily mean values
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array.

    :param data: 2D array of inflammation data
    :return: 1D array of daily max values
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array.

    :param data: 2D array of inflammation data
    :return: 1D array of daily min values
    """
    return np.min(data, axis=0)

def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    Negative values in the input data are replaced with 0.
    Nan values in the normalised data are replaced with 0.

    :param data: 2D array of inflammation data
    :return: 2D array of normalised patient data
    """
    data[data < 0] = 0
    max = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    return normalised