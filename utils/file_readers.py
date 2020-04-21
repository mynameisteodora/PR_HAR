"""
All file reading functions should be declared here
"""
from utils.constants import *
from utils.signal_processing import *

import tqdm


# reads a recording into a file and adds magnitude and PCA to it and removes the simple x, y, z axes
# with an option to remove the gyro
def read_single_path(path_to_csv, signal_smoothing_factor=10, keep_axes=False, downsample=True, header=7, gyro=False):
    """
    Function to read sensor data from a single file and apply simple transformation to its axes.
    You can optionally skip the gyroscope readings, as they are not required for MInf2.
    :param path_to_csv: Relative path to the data file
    :param signal_smoothing_factor: The size of the smoothing window. Default is 10.
    :param keep_axes: Set to True to keep the original axes (x, y and z)
    :param downsample: Set to True to downsample the signal from 25Hz to 12.5Hz
    :param header: The header of the csv file. For MInf2, the collected data has a header of 7.
    :param gyro: Set to True to enable gyroscope readings
    :return: Dataframe where columns are signal features and rows are points in time.
    """
    if header != 0:
        df = pd.read_csv(path_to_csv, header=header)
    else:
        names = ['local_ts', 'breathing_rate', 'breathing_signal', 'accel_x', 'accel_y', 'accel_z',
                 'bs_timestamp', 'rs_timestamp', 'activity_level', 'activity_type', 'current_exercise']
        df = pd.read_csv(path_to_csv, header=header, names=names)

    if downsample:
        df = df[::2]
        df.reset_index(inplace=True, drop=True)

    if not gyro and header != 0:
        df = df.drop(['gyro_x', 'gyro_y', 'gyro_z'], axis=1)

    # compute magnitudes
    df['accel_magnitude'] = np.sqrt(df['accel_x'] ** 2 + df['accel_y'] ** 2 + df['accel_z'] ** 2)

    if gyro:
        df['gyro_magnitude'] = np.sqrt(df['gyro_x'] ** 2 + df['gyro_y'] ** 2 + df['gyro_z'] ** 2)

    # compute first PC
    df['accel_pca'] = apply_pca(df, number_components=1, features='accel')['PC_1']

    if gyro:
        df['gyro_pca'] = apply_pca(df, number_components=1, features='gyro')['PC_1']

    if (keep_axes == False):
        if gyro:
            df = df.drop(['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'], axis=1)
        else:
            df = df.drop(['accel_x', 'accel_y', 'accel_z'], axis=1)

    # Add the high and low pass values here
    for col in df.columns:
        if ('accel' in col) or ('gyro' in col):
            new_col_smooth = col + "_smooth"
            new_col_lp = col + "_lp"
            new_col_hp = col + "_hp"
            new_col_grad = col + "_grad"
            new_col_double_grad = col + "_doublegrad"
            df[new_col_smooth] = smooth(signal_smoothing_factor, list(df[col]))

            # for low pass center the resulting signal
            lp_signal = low_pass(df, col)
            begin = len(lp_signal) - len(df[col])
            start = int(begin / 2)
            finish = len(lp_signal) - start
            df[new_col_lp] = lp_signal[start:finish]

            # high pass with cutoff = 0.3 (martins dizz)
            hp_signal = butter_filter(df, col)
            df[new_col_hp] = hp_signal

            # gradient
            grad = np.gradient(df[col])
            df[new_col_grad] = grad

            # double gradient
            doublegrad = np.gradient(df[new_col_grad])
            df[new_col_double_grad] = doublegrad

    return df


def read_by_paths(path_list):
    """
    Function for reading all files in a path list.
    :param path_list: A list of csv recordings we want to read
    :return: Dataframe of all recordings from the specified path list
    """
    # create empty df to concatenate to
    base_df = pd.DataFrame(data=None, columns=['timestamp', 'seq', 'accel_x', 'accel_y', 'accel_z', 'accel_magnitude',
                                               'accel_pca', 'accel_x_smooth', 'accel_x_lp', 'accel_x_hp',
                                               'accel_x_grad', 'accel_x_doublegrad', 'accel_y_smooth', 'accel_y_lp',
                                               'accel_y_hp', 'accel_y_grad', 'accel_y_doublegrad', 'accel_z_smooth',
                                               'accel_z_lp', 'accel_z_hp', 'accel_z_grad', 'accel_z_doublegrad',
                                               'accel_magnitude_smooth', 'accel_magnitude_lp', 'accel_magnitude_hp',
                                               'accel_magnitude_grad', 'accel_magnitude_doublegrad',
                                               'accel_pca_smooth', 'accel_pca_lp', 'accel_pca_hp', 'accel_pca_grad',
                                               'accel_pca_doublegrad', 'subject', 'activity', 'correctness'])

    activity_name_dict = get_activity_name_dict()

    for path in tqdm.tqdm(path_list):
        subject, activity_name, correctness, _ = path.split('/')[-1].split('_')

        df = read_single_path(path, keep_axes=True)
        df['subject'] = subject
        df['activity'] = activity_name_dict[activity_name]
        df['correctness'] = correctness.lower()

        # concatenate to base
        base_df = pd.concat([base_df, df])
        base_df.reset_index(drop=True, inplace=True)

    return base_df


def read_all_files():
    """
    Function for reading all the files (all subjects, all activities, all modes correct/incorrect).
    :return: Dataframe of all recordings, with columns = [subject, activity, recording]
    """
    paths = get_all_recording_paths()

    return read_by_paths(paths)


def read_files_filtered(subjects='all', activities='all', modes='all'):
    """
    Function for reading files filtered by subject, activity and mode
    :param subjects:
    :param activities:
    :param modes:
    :return:
    """
    if subjects == 'all' and activities == 'all' and modes == 'all':
        return read_all_files()

    paths = get_recording_paths(subjects, activities, modes)

    return read_by_paths(paths)

def read_clean_files(subjects='all', activities='all', model='all'):
    """
    Function for reading clean files. The files already contain the engineered axes (magnitude, pca, lp, grad etc)
    :param subjects: list of subjects names or 'all'
    :param activities: list of activity labels or 'all'
    :param model: 'correct', 'incorrect' or 'all'
    :return: dataframe containing all relevant information
    """
