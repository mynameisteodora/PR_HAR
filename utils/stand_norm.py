from utils.file_readers import *
from utils.constants import *
from utils.signal_processing import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def remove_outliers(df, activities='all', subjects='all', correctness='all',
                    axes=['accel_x', 'accel_y', 'accel_z']):
    """
    Given a dataframe, a list of activities, subjects and the type of correctness, this function removes outliers
    based on simple mean and std rules.
    :param df: Dataframe containing accelerometer, activity, subject and correctness data
    :param activities: A list of activities or 'all'
    :param subjects: A list of subject names or 'all'
    :param correctness: 'correct', 'incorrect' or 'all'
    :param axes: Columns of the df to compute the mean and std on.
    :return: New dataframe, means and std by (correctness) and activity.
    If only one correctness type is selected, the means and stds dictionaries
    will only have one level.
    """

    means_by_act = {}
    std_by_act = {}

    if activities == 'all':
        activities = range(10)

    if subjects == 'all':
        subjects = get_subject_names()
    else:
        mask_subj = (df['subject'] == subjects[0])
        for subj in subjects:
            mask_subj = mask_subj | (df['subject'] == subj)

        df = df[mask_subj]
        df.reset_index(drop=True, inplace=True)

    if correctness == 'all':
        correctness_list = ['correct', 'incorrect']

        # initialise dictionaries
        means_by_act['correct'] = {}
        means_by_act['incorrect'] = {}
        std_by_act['correct'] = {}
        std_by_act['incorrect'] = {}

        # initialise nested dictionaries
        for i in range(10):
            means_by_act['correct'][i] = []
            means_by_act['incorrect'][i] = []
            std_by_act['correct'][i] = []
            std_by_act['incorrect'][i] = []
    else:
        correctness_list = [correctness]

        # initialise dictionaries
        for i in range(10):
            means_by_act[i] = []
            std_by_act[i] = []

    idx_to_exclude_list = set()

    og_len = len(df)
    print("Original dataframe length = {}".format(og_len))

    for corr in correctness_list:

        for activity in activities:

            for col in axes:

                mask = (df['correctness'] == corr) & (df['activity'] == activity)

                mean = df[mask][col].mean()
                std = df[mask][col].std()

                # add stats to dictionaries
                if correctness == 'all':
                    means_by_act[corr][activity].append(mean)
                    std_by_act[corr][activity].append(std)
                else:
                    means_by_act[activity].append(mean)
                    std_by_act[activity].append(std)

                threshold_low = mean - 3 * std
                threshold_high = mean + 3 * std
                mask_outlier = (df[mask][col] >= threshold_low) & (df[mask][col] <= threshold_high)

                # take all initial indices and subtract the indices remaining after filtering.
                # their difference will tell you the indices that need removing
                initial_idx = set(df.index[mask].tolist())
                idx_remaining = set(df.index[mask & mask_outlier].tolist())
                idx_to_exclude_list = idx_to_exclude_list.union(initial_idx - idx_remaining)

    all_indices = set(df.index.tolist())
    remaining_indices = all_indices - idx_to_exclude_list

    # remove the indices from the original dataframe
    df = df.iloc[list(remaining_indices)]
    df.reset_index(drop=True, inplace=True)

    new_len = len(df)
    print("New dataframe length = {}".format(new_len))
    print("Removed {} outliers".format(og_len - new_len))

    return df, means_by_act, std_by_act


def standardise_data(df, activities='all', subjects='all', correctness='all',
                     axes=['accel_x', 'accel_y', 'accel_z', 'accel_magnitude', 'accel_pca']):
    """
    Standardises the data for each activity, subject and correctness.
    :param df:
    :param activities:
    :param subjects:
    :param correctness:
    :param axes:
    :return:
    """

    if activities == 'all':
        activities = range(10)

    if subjects == 'all':
        subjects = get_subject_names()
    else:
        mask_subj = (df['subject'] == subjects[0])
        for subj in subjects:
            mask_subj = mask_subj | (df['subject'] == subj)

        df = df[mask_subj]
        df.reset_index(drop=True, inplace=True)

    if correctness == 'all':
        correctness_list = ['correct', 'incorrect']
    else:
        correctness_list = [correctness]

    for correctness in ['correct', 'incorrect']:

        # filter by correctness
        mask_df_cor = df['correctness'] == correctness
        df_cor = df[mask_df_cor]
        # print("Correctness = {}\t\tSamples = {}".format(correctness, len(df_cor)))

        for activity in activities:

            # filter by act
            mask_act = df_cor['activity'] == activity
            df_act = df_cor[mask_act]
            # print("Activity = {}\t\tSamples = {}".format(activity, len(df_act)))

            for subject in subjects:
                # filter by subject
                mask_subj = df_act['subject'] == subject
                df_subj = df_act[mask_subj]
                # print("Subject = {}\t\tSamples = {}".format(subject, len(df_subj)))

                scaler = StandardScaler()
                scaler.fit(df_subj[axes])
                new_vals = scaler.transform(df_subj[axes])

                final_mask = mask_df_cor & (df['activity'] == activity) & (df['subject'] == subject)
                df.loc[final_mask, 'accel_x_standardised'] = new_vals[:, 0]
                df.loc[final_mask, 'accel_z_standardised'] = new_vals[:, 1]
                df.loc[final_mask, 'accel_y_standardised'] = new_vals[:, 2]
                df.loc[final_mask, 'accel_magnitude_standardised'] = new_vals[:, 3]
                df.loc[final_mask, 'accel_pca_standardised'] = new_vals[:, 4]

    return df


def normalise_data(df, activities='all', subjects='all', correctness='all',
                   axes=['accel_x_standardised', 'accel_y_standardised', 'accel_z_standardised',
                         'accel_magnitude_standardised', 'accel_pca_standardised'],
                   scaler_fit=None):
    """
    Normalises data
    :param df:
    :param activities:
    :param subjects:
    :param correctness:
    :param axes:
    :return:
    """

    scalers = {}

    if activities == 'all':
        activities = range(10)

    if subjects == 'all':
        subjects = get_subject_names()
    else:
        mask_subj = (df['subject'] == subjects[0])
        for subj in subjects:
            mask_subj = mask_subj | (df['subject'] == subj)

        df = df[mask_subj]
        df.reset_index(drop=True, inplace=True)

    if correctness == 'all':
        correctness_list = ['correct', 'incorrect']
    else:
        correctness_list = [correctness]

    for cor in correctness_list:
        scalers[cor] = {}

    for correctness in correctness_list:

        # filter by correctness
        mask_df_cor = df['correctness'] == correctness
        df_cor = df[mask_df_cor]
        # print("Correctness = {}\t\tSamples = {}".format(correctness, len(df_cor)))

        for activity in range(10):
            # filter by act
            mask_act = df_cor['activity'] == activity
            df_act = df_cor[mask_act]
            # print("Activity = {}\t\tSamples = {}".format(activity, len(df_act)))

            if scaler_fit is None:
                scaler = MinMaxScaler()
                scaler.fit(df_act[axes])
            else:
                scaler = scaler_fit[correctness][activity]

            new_vals = scaler.transform(df_act[axes])

            scalers[correctness][activity] = scaler

            final_mask = mask_df_cor & mask_act
            df.loc[final_mask, 'accel_x_normalised'] = new_vals[:, 0]
            df.loc[final_mask, 'accel_y_normalised'] = new_vals[:, 1]
            df.loc[final_mask, 'accel_z_normalised'] = new_vals[:, 2]
            df.loc[final_mask, 'accel_magnitude_normalised'] = new_vals[:, 3]
            df.loc[final_mask, 'accel_pca_normalised'] = new_vals[:, 4]

    return df, scalers


def normalise_data(df, activities='all', subjects='all', correctness='all',
                   axes=['accel_x_standardised', 'accel_y_standardised', 'accel_z_standardised',
                         'accel_magnitude_standardised', 'accel_pca_standardised'],
                   scaler_fit=None):
    """
    Normalises data
    :param df:
    :param activities:
    :param subjects:
    :param correctness:
    :param axes:
    :return:
    """

    scalers = {}

    if activities == 'all':
        activities = range(10)

    if subjects == 'all':
        subjects = get_subject_names()
    else:
        mask_subj = (df['subject'] == subjects[0])
        for subj in subjects:
            mask_subj = mask_subj | (df['subject'] == subj)

        df_subj = df[mask_subj]
        df_subj.reset_index(drop=True, inplace=True)

    if correctness == 'all':
        correctness_list = ['correct', 'incorrect']
    else:
        correctness_list = [correctness]

    for cor in correctness_list:
        scalers[cor] = {}

    for correctness in correctness_list:

        # filter by correctness
        mask_df_cor = df_subj['correctness'] == correctness
        df_cor = df_subj[mask_df_cor]
        # print("Correctness = {}\t\tSamples = {}".format(correctness, len(df_cor)))

        for activity in range(10):
            # filter by act
            mask_act = df_cor['activity'] == activity
            df_act = df_cor[mask_act]
            # print("Activity = {}\t\tSamples = {}".format(activity, len(df_act)))

            if scaler_fit is None:
                scaler = MinMaxScaler()
                scaler.fit(df_act[axes])
            else:
                scaler = scaler_fit[correctness][activity]

            new_vals = scaler.transform(df_act[axes])

            scalers[correctness][activity] = scaler

            final_mask = mask_df_cor & mask_act
            df_subj.loc[final_mask, 'accel_x_normalised'] = new_vals[:, 0]
            df_subj.loc[final_mask, 'accel_y_normalised'] = new_vals[:, 1]
            df_subj.loc[final_mask, 'accel_z_normalised'] = new_vals[:, 2]
            df_subj.loc[final_mask, 'accel_magnitude_normalised'] = new_vals[:, 3]
            df_subj.loc[final_mask, 'accel_pca_normalised'] = new_vals[:, 4]

    return df_subj, scalers


def normalise_data_universal(df, activities='all', subjects='all', correctness='all',
                             axes=['accel_x_standardised', 'accel_y_standardised', 'accel_z_standardised',
                                   'accel_magnitude_standardised', 'accel_pca_standardised'],
                             scaler_fit=None):
    """
    Uses a univsersal scaler to normalise all data, regardless of subject or activity
    The only variable is speed
    :param df:
    :param activities:
    :param subjects:
    :param correctness:
    :param axes:
    :return:
    """

    scalers = {}

    if activities == 'all':
        activities = range(10)

    if subjects == 'all':
        subjects = get_subject_names()
    else:
        mask_subj = (df['subject'] == subjects[0])
        for subj in subjects:
            mask_subj = mask_subj | (df['subject'] == subj)

        df_subj = df[mask_subj]
        df_subj.reset_index(drop=True, inplace=True)

    if correctness == 'all':
        correctness_list = ['correct', 'incorrect']
    else:
        correctness_list = [correctness]

    for cor in correctness_list:
        scalers[cor] = {}

    for correctness in correctness_list:

        # filter by correctness
        mask_df_cor = df_subj['correctness'] == correctness
        df_cor = df_subj[mask_df_cor]
        # print("Correctness = {}\t\tSamples = {}".format(correctness, len(df_cor)))

        if scaler_fit is None:
            scaler = MinMaxScaler()
            scaler.fit(df_cor[axes])
        else:
            scaler = scaler_fit[correctness]

        new_vals = scaler.transform(df_cor[axes])
        scalers[correctness] = scaler

        df_subj.loc[mask_df_cor, 'accel_x_normalised'] = new_vals[:, 0]
        df_subj.loc[mask_df_cor, 'accel_y_normalised'] = new_vals[:, 1]
        df_subj.loc[mask_df_cor, 'accel_z_normalised'] = new_vals[:, 2]
        df_subj.loc[mask_df_cor, 'accel_magnitude_normalised'] = new_vals[:, 3]
        df_subj.loc[mask_df_cor, 'accel_pca_normalised'] = new_vals[:, 4]

    return df_subj, scalers


def cross_normalise_data(df, scaler_fit, activity_norm=0, subjects='all', correctness='all',
                         axes=['accel_x_standardised', 'accel_y_standardised', 'accel_z_standardised',
                               'accel_magnitude_standardised', 'accel_pca_standardised']):
    """
    Takes a dataset and a fit normaliser ONLY FOR ONE ACTIVITY
    and returns the dataset normalised with that specific normaliser
    :param df:
    :param activities:
    :param subjects:
    :param correctness:
    :param axes:
    :return:
    """
    # prefilter to erase the activity fit for the normaliser
    df = df[df['activity'] != activity_norm]

    if subjects == 'all':
        subjects = get_subject_names()
        df_subj = df
    else:
        mask_subj = (df['subject'] == subjects[0])
        for subj in subjects:
            mask_subj = mask_subj | (df['subject'] == subj)

        df_subj = df[mask_subj]
        df_subj.reset_index(drop=True, inplace=True)

    if correctness == 'all':
        correctness_list = ['correct', 'incorrect']
    else:
        correctness_list = [correctness]

    for correctness in correctness_list:

        # filter by correctness
        mask_df_cor = df_subj['correctness'] == correctness
        df_cor = df_subj[mask_df_cor]
        # print("Correctness = {}\t\tSamples = {}".format(correctness, len(df_cor)))

        # load the scaler for the specific selected activity
        print("Loading scaler for correctness = {} and activity = {}".format(correctness, activity_norm))
        scaler = scaler_fit[correctness][activity_norm]

        for activity in range(10):
            if activity != activity_norm:
                # filter by act
                mask_act = df_cor['activity'] == activity
                df_act = df_cor[mask_act]
                # print("Activity = {}\t\tSamples = {}".format(activity, len(df_act)))

                scaler.fit(df_act[axes])

                new_vals = scaler.transform(df_act[axes])

                final_mask = mask_df_cor & mask_act
                df_subj.loc[final_mask, 'accel_x_normalised'] = new_vals[:, 0]
                df_subj.loc[final_mask, 'accel_y_normalised'] = new_vals[:, 1]
                df_subj.loc[final_mask, 'accel_z_normalised'] = new_vals[:, 2]
                df_subj.loc[final_mask, 'accel_magnitude_normalised'] = new_vals[:, 3]
                df_subj.loc[final_mask, 'accel_pca_normalised'] = new_vals[:, 4]

    return df_subj


def normalise_data_single(df, activities='all', subjects='all', correctness='all',
                          axes=['accel_x_standardised', 'accel_y_standardised', 'accel_z_standardised',
                                'accel_magnitude_standardised', 'accel_pca_standardised']):
    if activities == 'all':
        activities = range(10)

    if subjects == 'all':
        subjects = get_subject_names()
    else:
        mask_subj = (df['subject'] == subjects[0])
        for subj in subjects:
            mask_subj = mask_subj | (df['subject'] == subj)

        df_subj = df[mask_subj]
        df_subj.reset_index(drop=True, inplace=True)

    if correctness == 'all':
        correctness_list = ['correct', 'incorrect']
    else:
        correctness_list = [correctness]

    for correctness in correctness_list:
        # filter by correctness
        mask_df_cor = df_subj['correctness'] == correctness
        df_cor = df_subj[mask_df_cor]
        # print("Correctness = {}\t\tSamples = {}".format(correctness, len(df_cor)))

        for act in activities:
            for subj in subjects:
                scaler = MinMaxScaler()

                mask_df_filtered = (df_cor['activity'] == act) & (df_cor['subject'] == subj)
                df_filtered = df_cor[mask_df_filtered]

                scaler.fit(df_filtered[axes])
                new_vals = scaler.transform(df_filtered[axes])

                final_mask = mask_df_cor & mask_df_filtered

                df_subj.loc[final_mask, 'accel_x_normalised'] = new_vals[:, 0]
                df_subj.loc[final_mask, 'accel_y_normalised'] = new_vals[:, 1]
                df_subj.loc[final_mask, 'accel_z_normalised'] = new_vals[:, 2]
                df_subj.loc[final_mask, 'accel_magnitude_normalised'] = new_vals[:, 3]
                df_subj.loc[final_mask, 'accel_pca_normalised'] = new_vals[:, 4]

    return df_subj
