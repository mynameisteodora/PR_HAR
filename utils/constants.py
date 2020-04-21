"""
A file where all the commonly used constants should be declared
"""
import os
import glob

DATA_FOLDER = '../Data'
CLEAN_DATA_FOLDER = "../Clean_Data"
PLOTS_FOLDER = "../Plots"

EXPERIMENT_STATS = "{}/experiment_stats.csv".format(PLOTS_FOLDER)
ALL_EXPERIMENT_STATS = "{}/all_experiment_stats.csv".format(PLOTS_FOLDER)


def get_activity_label_dict():
    """
    Returns the dictionary of activities
    :return: Dictionary with activity labels as keys and activity names as values
    """
    activity_label_dict = {
        0: 'Sit to stand',
        1: 'Knee extension',
        2: 'Squats',
        3: 'Heel raises',
        4: 'Bicep curl',
        5: 'Shoulder press',
        6: 'Wall push offs',
        7: 'Leg slide',
        8: 'Step ups',
        9: 'Walking'
    }
    return activity_label_dict


def get_activity_name_dict():
    """
    Returns the dictionary of activity names
    :return: Dictionary with activity names as keys and activity labels as values
    """
    activity_name_dict = {
        'Sit to stand': 0,
        'Knee extension': 1,
        'Squats': 2,
        'Heel raises': 3,
        'Bicep curl': 4,
        'Shoulder press': 5,
        'Wall push offs': 6,
        'Leg slide': 7,
        'Step ups': 8,
        'Walking': 9
    }

    return activity_name_dict


def get_subject_names():
    """
    Returns the list of available subject names, assuming they are placed in the 'Data' folder
    :return: List of subject names
    """
    subject_names = os.listdir(CLEAN_DATA_FOLDER)
    if '.DS_Store' in subject_names:
        subject_names.remove('.DS_Store')

    return sorted(subject_names)


def get_all_recording_paths():
    """
    Returns a list of paths for all available recordings
    :return: List of paths
    """
    file_list = glob.glob(DATA_FOLDER + "**/**/*.csv", recursive=True)
    return file_list


def get_recording_paths(subjects='all', activities='all', modes='all'):
    """
    Returns a list of paths for the specified subjects, activities and modes
    :param subjects: 'all' or a list of subject names
    :param activities: 'all' or a list of activity labels
    :param modes: 'all', 'correct' or 'incorrect'
    :return: List of file paths
    """
    if subjects == 'all' and activities == 'all' and modes == 'all':
        return get_all_recording_paths()

    # first filter by mode
    if modes == 'all':
        modes_filter = '**'
    elif modes == 'correct':
        modes_filter = 'correct'
    else:
        modes_filter = 'incorrect'

    subject_paths = []

    if type(subjects) == list:
        # custom subjects
        for subject_filter in subjects:
            subject_paths.extend(glob.glob('{}/{}/{}/*.csv'.format(DATA_FOLDER, subject_filter, modes_filter), recursive=True))

    else:
        subject_filter = '**'
        # all subjects
        subject_paths.extend(glob.glob('{}/{}/{}/*.csv'.format(DATA_FOLDER, subject_filter, modes_filter), recursive=True))

    activities_paths = []
    activities_dict = get_activity_label_dict()

    if type(activities) == list:
        # custom activities
        # we have to split the name of each file to get the activity name
        for file_path in subject_paths:
            subject_name, activity_name, correctness, _ = file_path.split('_')

            for activity_label in activities:
                if activities_dict[activity_label] == activity_name:
                    activities_paths.append(file_path)

    else:
        # all activities
        activities_paths = subject_paths

    return activities_paths

