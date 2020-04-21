from utils.constants import *
from utils.file_readers import *
from scipy.stats import *


def generate_sequence(dataframe, target_vals, columns, n_time_steps, step):
    """
    Given a dataframe and the columns to keep for the sliding window, generates sliding, overlapping windows.
    :param dataframe: The source Pandas Dataframe
    :param target_vals: The dataframe column which contains the target values, in the form of df[col]
    :param columns: A list of column names to keep for the final windows (will be the number of channels)
    :param n_time_steps: The number of time samples in a window
    :param step: The length of the step (determines overlap)
    :return: Segments, a three dimensional array of shape (num_windows, num_original_columns, n_time_steps)
            Labels, a one dimensional array of shape (num_windows,)
    """
    segments = []
    labels = []

    for i in range(0, len(dataframe) - n_time_steps, step):
        col_slices = []
        for col in columns:
            col_slice = dataframe[col].values[i: i + n_time_steps]
            col_slices.append(col_slice)

        label = stats.mode(target_vals[i: i + n_time_steps])[0][0]
        segments.append(col_slices)
        labels.append(label)

    return segments, labels


def reshape_segments(x, y, n_time_steps, n_features):
    """
    Reshape input segments and one-hot encode labels
    :param x: Training/testing data points
    :param y: Training/testing labels
    :param n_time_steps: Number of time steps in a window
    :param n_features: Number of features per training example
    :return: x_reshaped, y_reshaped
    """
    x_reshaped = np.asarray(x, dtype=np.float32).reshape(-1, n_time_steps, n_features)
    y_reshaped = np.asarray(pd.get_dummies(y), dtype=np.float32)
    return x_reshaped, y_reshaped

def reshape_segments_clean(x, y, n_time_steps, n_features):
    print(np.shape(x))
    print(np.shape(y))
    x_reshaped = np.transpose(np.asarray(x, dtype=np.float32), axes=(0,2,1))
    y_reshaped = np.asarray(pd.get_dummies(y), dtype=np.float32)
    return x_reshaped, y_reshaped


def generate_dataset(df, n_time_steps, n_features, step,
                     features=['accel_x', 'accel_y', 'accel_z'],
                     one_vs_all_activity=0, subjects='all', correctness='all',
                     downsample=True, downsample_rate=2):
    """
    Create a training dataset from a dataframe, using specified sliding window dimensions.
    :param df: Dataframe containing raw data
    :param n_time_steps: Number of samples in a window
    :param n_features: Number of features (channels)
    :param step: Step size, determines window overlap. If step = n_time_steps/2 then overlap is 50%
    :param features: Name of features to include, type should be list
    :param one_vs_all_activity: Can be an integer (range(10)) or 'all'. Integers will generate one-vs-all data.
    'All' will generate one-hot encoded vectors as labels
    :param subjects: List of subject names or 'all'
    :param correctness: 'correct', 'incorrect' or 'all'
    :param downsample: True if we are performing one-vs-all and we want to downsample the negative class
    :param downsample_rate: Number of orders to downsample. For downsample_rate=n, we will take only every other nth
    training point from the negative glass.
    :return: X_train, y_train
    """

    num_act_true = 0
    num_act_false = 0

    if one_vs_all_activity == 'all':
        one_vs_all_activity = np.identity(10)
        downsample = False
        second_dim = 10
    else:
        # one_vs_all_activity = np.identity(2)
        second_dim = 2

    if type(subjects) == list:
        mask_subj = (df['subject'] == subjects[0])
        for subject in subjects:
            mask_subj = mask_subj | (df['subject'] == subject)

        df = df[mask_subj]

    else:
        subjects = get_subject_names()

    if correctness == 'all':
        correctness_list = ['correct', 'incorrect']
    elif correctness == 'correct':
        correctness_list = ['correct']
    else:
        correctness_list = ['incorrect']

    X_train = np.empty((0, n_time_steps, n_features))
    y_train = np.empty((0, second_dim))

    for correctness in correctness_list:

        for activity in range(10):

            if type(one_vs_all_activity) == int and activity == one_vs_all_activity:
                # TODO these might need to be reversed
                act_label = np.array([1, 0])
            elif type(one_vs_all_activity) == int:
                act_label = np.array([0, 1])
            elif type(one_vs_all_activity) == np.ndarray:
                act_label = one_vs_all_activity[activity]

            for subject in subjects:
                print(f"Generating sets for ACTIVITY {activity}, SUBJECT {subject}, CORR {correctness}")
                mask = (df['correctness'] == correctness) & (df['activity'] == activity) & (df['subject'] == subject)

                segments, labels = generate_sequence(dataframe=df[mask], target_vals=df[mask]['activity'],
                                                     columns=features, n_time_steps=n_time_steps, step=step)
                # reshape
                segments, labels = reshape_segments_clean(segments, labels, n_time_steps=n_time_steps, n_features=n_features)
                labels = np.full((segments.shape[0], second_dim), act_label)

                if type(act_label) == int and act_label == 0 and downsample:
                    # downsample
                    segments = segments[::downsample_rate]
                    labels = labels[::downsample_rate]

                    num_act_false += segments.shape[0]
                else:
                    num_act_true += segments.shape[0]

                X_train = np.concatenate((X_train, segments))
                y_train = np.concatenate((y_train, labels))

    print("Total samples for true activity = {}".format(num_act_true))
    print("Total samples for false activity = {}".format(num_act_false))
    return X_train, y_train