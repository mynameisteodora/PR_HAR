"""
Mostly plotting functions, comments should indicate which notebooks they are used in for examples.
"""
import matplotlib.pyplot as plt
from utils.constants import *


def plot_comparative_histograms(df, activity, subjects='all', correctness='correct',
                                hspace=1, wspace=0.5, y_suptitle=0.90, figsize=(15, 30)):
    """
    Plots comparative histograms for values of accel_x, y and z for selected subjects and activities.
    Look at notebook 01_Data exploration for example usage.
    :param df: Dataframe containing sensor data
    :param activity: Label of desired activity
    :param subjects: List of subject names or 'all'
    :param correctness: 'correct' or 'incorrect'
    :param hspace: Adjust the height spacing between subplots
    :param wspace: Adjust the width spacing between subplots
    :param y_suptitle: Adjust the figure title height
    :param figsize: Adjust the figure size
    :return: Figure object. Can use later to save the plot
    """
    # filter by subjects
    if type(subjects) == list:
        num_subjects = len(subjects)
        mask = (df['subject'] == subjects[0])
        for i in range(num_subjects):
            mask = mask | (df['subject'] == subjects[i])
        df = df[mask]
    else:
        subjects = get_subject_names()
        num_subjects = len(subjects)

    # filter by correctness
    mask = df['correctness'] == correctness
    df = df[mask]

    # filter by activity
    mask = df['activity'] == activity
    df = df[mask]

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    fig.suptitle('Histograms for activity {}, {}'.format(activity, correctness), fontsize=18, y=y_suptitle)
    fig.tight_layout()
    counter = 1
    subject_num = 0
    accel_axes = ['accel_x', 'accel_y', 'accel_z']
    for i in range(1, 3*num_subjects + 1):
        ax = fig.add_subplot(num_subjects, 3, i)
        mask = (df['subject'] == subjects[subject_num])
        plt.hist(df[mask][accel_axes[counter - 1]])

        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((0, 150))

        if i == 1:
            plt.title('accel_x')
        elif i == 2:
            plt.title('accel_y')
        elif i == 3:
            plt.title('accel_z')

        if counter == 1:
            ax.set_ylabel('Subject {}'.format(subject_num))

        counter += 1

        if counter == 4:
            counter = 1
            subject_num += 1
    fig.tight_layout()
    return fig


def plot_exercise_duration_boxplot(df, correctness, subjects='all'):
    """
    Plots a boxplot with the exercise durations taken from a subset of subjects.
    Look at notebook 01_Data exploration for example usage.
    :param df: Dataframe containing the desired data
    :param correctness: 'correct' or 'incorrect'
    :param subjects: List of subject names or 'all'
    :return: Figure object. Can use later to save the plot
    """
    if subjects == 'all':
        subjects = get_subject_names()

    activity_label_dict = get_activity_label_dict()

    activity_durations = {}
    for i in range(10):
        activity_durations[i] = []

    for i in range(10):
        # filter by activity and correctness
        mask_act = (df['activity'] == i) & (df['correctness'] == correctness)
        df_act = df[mask_act]

        for j in range(len(subjects)):
            mask = df_act['subject'] == subjects[j]
            df_len = df_act[mask]
            seconds = len(df_len) / 12.5
            activity_durations[i].append(seconds)

    act_durations_plot = []
    act_names = [activity_label_dict[i] for i in range(10)]
    for k, v in activity_durations.items():
        act_durations_plot.append(v)

    fig = plt.figure(figsize=(10, 6))
    plt.boxplot(act_durations_plot, labels=act_names)
    plt.xticks(rotation=45)
    plt.ylabel('Seconds')
    plt.ylim((0, 100))
    plt.title('Average {} exercise duration'.format(correctness), fontsize=16)
    plt.tight_layout()
    return fig