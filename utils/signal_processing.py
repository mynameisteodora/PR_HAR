"""
Functions for processing raw signal
"""
import numpy as np
import pandas as pd
import scipy.fftpack
from scipy.signal import hilbert, butter, lfilter, filtfilt
from sklearn.decomposition import PCA

def low_pass(dataframe, column):
    """
    A function for low-passing the column of a dataframe
    :param dataframe: Source Pandas dataframe
    :param column: Dataframe column to low-pass
    :return: Low-pass signal, type np.array
    """
    fc = 0.08
    b = 0.08
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1
    n = np.arange(N)

    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = 0.7 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    s = list(dataframe[column])
    new_signal = np.convolve(s, sinc_func)

    return new_signal

def butter_filter(dataframe, column, sampl_freq=12.5, cutoff_freq=0.3, filter_order=1, filter_type='high'):
    """
    A function for applying a Butterworth filter to the column of a dataframe.
    Used for extracting the linear (body) acceleration from raw accelerometer data
    :param dataframe: Pandas dataframe
    :param column: Dataframe column
    :param sampl_freq: Signal sampling frequency
    :param cutoff_freq: Cutoff frequency. Default is 0.3 for high-pass filter, to extract the body acceleration
    :param filter_order: Order of the filter. Default is 1
    :param filter_type: 'high' or 'low'
    :return: Filtered signal, list
    """
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = sampl_freq
    fc = cutoff_freq  # Cut-off frequency of the filter
    nyq = 0.5 * fs

    Wn = fc / nyq
    # print(Wn)

    N = filter_order  # Order of the filter
    # Wn = 20
    b, a = butter(N, Wn, btype=filter_type, analog=False)

    new_signal = filtfilt(b, a, dataframe[column])

    return new_signal

def find_peaks(column):
    """
    Given a filtered signal, counts the number of peaks and troughs in an array
    :param column: An array of numbers, extracted from a dataframe
    :return: The number of peaks
    """
    peaks = 0
    for i in range(1, len(column) - 1):
        if (column[i] > column[i - 1] and column[i] > column[i + 1]):
            peaks += 1
        elif (column[i] < column[i - 1] and column[i] < column[i + 1]):
            peaks += 1
    return peaks

def smooth(window_width, array):
    """
    Sliding average smoothing.
    :param window_width: The width of the average window, in datapoints
    :param array: An array of numbers, extracted from a dataframe
    :return: Smoothed signal, as a np.array
    """
    cumsum_vec = np.cumsum(array)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    ma_vec = np.append(ma_vec, array[len(array) - window_width:])
    return ma_vec


def apply_pca(dataframe, number_components=1, features='accel'):
    """
    Extracting the first principal component out of a 3-dimensional signal (from axes x, y and z)
    :param dataframe: Pandas dataframe
    :param number_components: The number of principal components to extract. Default is 1
    :param features: Type of features to apply PCA to. The options are ['accel', 'gyro', 'accel_smooth', 'gyro_smooth', 'all']
    :return: PCA-decomposed dataframe
    """
    # This assumes you are passing the dataframe right after reading
    features_accel = ['accel_x', 'accel_y', 'accel_z']
    features_gyro = ['gyro_x', 'gyro_y', 'gyro_z']
    features_accel_smooth = ['accel_x_smooth', 'accel_y_smooth', 'accel_z_smooth']
    features_gyro_smooth = ['gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth']
    features_all = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                    'accel_magnitude', 'gyro_magnitude', 'accel_x_smooth', 'accel_y_smooth',
                    'accel_z_smooth', 'gyro_x_smooth', 'gyro_y_smooth', 'gyro_z_smooth',
                    'accel_magnitude_smooth', 'gyro_magnitude_smooth']
    features_accel_standardised = ['accel_x_standardised', 'accel_y_standardised', 'accel_z_standardised']
    features_accel_normalised = ['accel_x_normalised', 'accel_y_normalised', 'accel_z_normalised']

    new_cols = []
    for i in range(1, number_components + 1):
        new_cols.append("PC_" + str(i))

    # Separating out the features
    if (features == 'accel'):
        X = dataframe.loc[:, features_accel].values
    elif (features == 'gyro'):
        X = dataframe.loc[:, features_gyro].values
    elif (features == 'accel_smooth'):
        X = dataframe.loc[:, features_accel_smooth].values
    elif (features == 'gyro_smooth'):
        X = dataframe.loc[:, features_gyro_smooth].values
    elif (features == 'all'):
        X = dataframe.loc[:, features_all].values
    elif (features == 'standardised'):
        X = dataframe.loc[:, features_accel_standardised].values
    elif (featured == 'normalised'):
        X = dataframe.loc[:, features_accel_normalised].values
    # Separating out the target
    # y = dataframe.loc[:,['class']].values
    # Standardizing the features
    # X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=number_components)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=new_cols)
    # print(pca.explained_variance_ratio_)
    return principalDf

# This function is taken from an utility pack sent by Darius
def fourier_spectrum(signal, sampling_frequency=25):
    # Number of samplepoints
    n = len(signal)

    y = scipy.fftpack.fft(signal)
    x = np.linspace(0.0, sampling_frequency / 2.0, n / 2.0)

    # Only take positive frequencies
    y_pos = 2.0 / n * np.abs(y[:int(n / 2.0)])
    return x, y_pos