"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
import glob
import os

def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    :return: 2D Numpy array of data
    """
    return np.loadtxt(fname=filename, delimiter=',')

def load_json(filename):
    """Load a numpy array from a JSON document.
    
    Expected format:
    [
      {
        "observations": [0, 1]
      },
      {
        "observations": [0, 2]
      }    
    ]
    :param filename: Filename of CSV to load
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]

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
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a Numpy array")
    if len(data.shape) != 2:
        raise ValueError("Input data must be a 2D array")
    if np.any(data < 0):
        raise ValueError("Negative values not allowed in input data")

    max_values = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_values[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    return normalised

def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    
    data = data_source.load_inflammation_data()
    daily_standard_deviation = compute_standard_deviation(data)


    return daily_standard_deviation

def compute_standard_deviation(data):
    """Computes the standard deviation by day between datasets.
    param data: list of 2D numpy arrays, each array is a dataset of inflammation data.
    return: 1D numpy array of standard deviation by day between datasets.
    """
    means_by_day = map(daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))
    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    return daily_standard_deviation

class CSVDataSource:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_inflammation_data(self):
        """
        Loads all inflammation data from CSV files in the specified directory.
        Returns a list of 2D numpy arrays, each array is a CSV file of inflammation data.
        """
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(load_csv, data_file_paths)
        return list(data)

class JSONDataSource:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_inflammation_data(self):
        """
        Loads all inflammation data from JSON files in the specified directory.
        Returns a list of 2D numpy arrays, each array is a JSON file of inflammation data.
        """
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data JSON files found in path {self.data_dir}")
        data = map(load_json, data_file_paths)
        return list(data)